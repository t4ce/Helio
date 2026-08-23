//! Closed, typed access to application-owned SceneDB components and subsystem
//! payloads.
//!
//! The high-level [`Scene`](super::Scene) deliberately does not expose its
//! `SceneAuthority` or SceneDB `World`. Applications may register GPU columns
//! during construction, then reach only their own component and subsystem
//! types through the facades in this module. An opaque
//! [`SceneExtensionEntity`] prevents component mutations from being aimed at
//! Helio's built-in entities. Application subsystems live in one private,
//! SceneDB-registered type map and receive no raw World or phase hooks.

use std::any::{type_name, Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use helio_scenedb::{
    component_id, GrowableGpuColumnSet, MirrorMode, SceneAuthority, SceneDbComponent,
    SceneGpuStore,
};

/// SceneDB's copyable DirtyTracked storage marker, re-exported under the
/// extension-facing name. Implement this together with [`SceneDataComponent`];
/// [`SceneDirtyTrackedComponent`] is then provided automatically.
pub use helio_scenedb::MutableGpuComponent as SceneDirtyTrackedStorage;

use super::actor::SceneActorId;
use super::Scene;

/// Common marker for application-owned SceneDB components.
///
/// Implement this only for types owned by the application. Helio's built-in
/// component types deliberately do not implement it; Rust's orphan rules stop
/// downstream crates from opting those foreign types into this facade.
///
/// ```compile_fail
/// use helio::{SceneCpuComponent, SceneDataComponent, SceneSpriteRow};
///
/// // Both the trait and this Helio-owned type are foreign to the application.
/// impl SceneDataComponent for SceneSpriteRow {}
/// impl SceneCpuComponent for SceneSpriteRow {}
/// ```
pub trait SceneDataComponent: SceneDbComponent {}

/// Marker for an application component with no `#[gpu]` fields.
///
/// Updates through [`SceneDataMut::edit_cpu`] borrow the canonical value in
/// place. The facade reflects the component on first use and rejects a type
/// that actually contains GPU-partnered fields.
pub trait SceneCpuComponent: SceneDataComponent {}

/// Marker for a `Copy` application component whose `#[gpu]` fields are all
/// `DirtyTracked` (the default mode of bare `#[gpu]`).
///
/// Updates use a transactional copy/edit/reinsert so SceneDB's differential
/// mirror dispatch observes every changed GPU field.
pub trait SceneDirtyTrackedComponent: SceneDataComponent + SceneDirtyTrackedStorage {}

impl<T> SceneDirtyTrackedComponent for T where T: SceneDataComponent + SceneDirtyTrackedStorage {}

/// Marker for an application component whose `#[gpu]` fields are all
/// `#[gpu(mirror = Once)]`.
///
/// Once components can be added only when absent. To start a new handoff
/// lifetime, remove the component and add it again explicitly.
pub trait SceneOnceComponent: SceneDataComponent {}

/// Marker for an application component combining DirtyTracked and Once GPU
/// fields. Mixed components intentionally have no whole-component edit API:
/// remove and add again to begin a new presence/Once lifetime.
pub trait SceneMixedComponent: SceneDataComponent {}

/// Marker for an application-owned global/index/query payload stored in
/// SceneDB's private extension-subsystem container.
///
/// This is intentionally not SceneDB's phase-aware `Subsystem` trait. Payloads
/// receive no World, GPU-store, or phase-hook access; application execution
/// remains on Helio's actor/executor side. Implement the marker only for types
/// owned by the application. Rust's orphan rules prevent downstream crates
/// from using this facade to reach Helio's built-in subsystem or component
/// types.
///
/// ```compile_fail
/// use helio::{SceneDataSubsystem, SceneSpriteRow};
///
/// // Both the marker and this Helio-owned type are foreign to the application.
/// impl SceneDataSubsystem for SceneSpriteRow {}
/// ```
pub trait SceneDataSubsystem: Any + Send + Sync + 'static {}

#[derive(Default)]
pub(in crate::scene) struct ExtensionSubsystemStore {
    payloads: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl ExtensionSubsystemStore {
    fn get<T: SceneDataSubsystem>(&self) -> Option<&T> {
        self.payloads
            .get(&TypeId::of::<T>())
            .and_then(|payload| payload.downcast_ref::<T>())
    }

    fn get_mut<T: SceneDataSubsystem>(&mut self) -> Option<&mut T> {
        self.payloads
            .get_mut(&TypeId::of::<T>())
            .and_then(|payload| payload.downcast_mut::<T>())
    }

    fn insert<T: SceneDataSubsystem>(&mut self, value: T) -> bool {
        let type_id = TypeId::of::<T>();
        if self.payloads.contains_key(&type_id) {
            return false;
        }
        self.payloads.insert(type_id, Box::new(value));
        true
    }

    fn remove<T: SceneDataSubsystem>(&mut self) -> Option<T> {
        self.payloads
            .remove(&TypeId::of::<T>())
            .and_then(|payload| payload.downcast::<T>().ok())
            .map(|payload| *payload)
    }

    fn clear(&mut self) {
        self.payloads.clear();
    }
}

impl helio_scenedb::Subsystem for ExtensionSubsystemStore {
    fn name(&self) -> &'static str {
        "helio.scene.extension_subsystems"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Opaque construction-time capability for registering application-owned GPU
/// component partners.
///
/// It deliberately exposes only typed growable-World registration. In
/// particular, it provides neither access to SceneDB's GPU store nor a way to
/// inspect, mutate, or retain Helio's already-registered built-in buffers.
///
/// ```compile_fail
/// fn cannot_reach_the_store(registrar: &mut helio::SceneComponentRegistrar<'_>) {
///     let _ = registrar.gpu_buffer_snapshot_for_key("helio.scene.lights");
/// }
/// ```
pub struct SceneComponentRegistrar<'a> {
    store: &'a mut SceneGpuStore,
    device: &'a Arc<wgpu::Device>,
}

impl<'a> SceneComponentRegistrar<'a> {
    pub(in crate::scene) fn new(
        store: &'a mut SceneGpuStore,
        device: &'a Arc<wgpu::Device>,
    ) -> Self {
        Self { store, device }
    }

    /// Register one derived application component before SceneDB's World
    /// mirror is attached. `initial_capacity` is a cheap starting allocation;
    /// the component-local buffers grow transparently when later rows exceed it.
    pub fn register<T>(&mut self, initial_capacity: u32)
    where
        T: SceneDataComponent + GrowableGpuColumnSet,
    {
        <T as GrowableGpuColumnSet>::register_gpu_columns_growable(
            self.store,
            initial_capacity.max(1),
            self.device,
        );
    }
}

/// Opaque, generation-bearing identity for an application-owned SceneDB
/// entity. There is intentionally no conversion to or from Helio resource
/// handles or SceneDB's raw `Entity`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneExtensionEntity(pub(super) helio_scenedb::Entity);

impl fmt::Debug for SceneExtensionEntity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SceneExtensionEntity(..)")
    }
}

/// Allocation-epoch-aware read-only GPU partner publication.
#[derive(Debug)]
pub struct SceneDataBuffer {
    pub buffer: wgpu::Buffer,
    pub epoch: u64,
    pub row_stride: u32,
}

/// Errors from the application-owned SceneDB facade.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SceneDataError {
    #[error("scene extension entity is stale or does not belong to the extension domain")]
    InvalidEntity,
    #[error("component {component} is already present on the extension entity")]
    ComponentAlreadyPresent { component: &'static str },
    #[error("component {component} is not present on the extension entity")]
    ComponentMissing { component: &'static str },
    #[error("application subsystem {subsystem} is already present")]
    SubsystemAlreadyPresent { subsystem: &'static str },
    #[error("application subsystem {subsystem} is not present")]
    SubsystemMissing { subsystem: &'static str },
    #[error(
        "component {component} has SceneDB GPU policy {actual}, but the facade requires {expected}"
    )]
    ComponentPolicyMismatch {
        component: &'static str,
        expected: &'static str,
        actual: &'static str,
    },
    #[error(
        "GPU columns for component {component} were not registered in Scene::new_with_component_registration"
    )]
    GpuColumnsNotRegistered { component: &'static str },
}

/// Read-only access to application-owned SceneDB data.
pub struct SceneDataView<'a> {
    authority: &'a SceneAuthority,
    subsystems: &'a ExtensionSubsystemStore,
}

impl<'a> SceneDataView<'a> {
    pub(super) fn new(authority: &'a SceneAuthority) -> Self {
        let subsystems = authority
            .subsystem::<ExtensionSubsystemStore>()
            .expect("extension subsystem store is registered at Scene construction");
        Self {
            authority,
            subsystems,
        }
    }

    /// Read one application-owned global/index/query payload.
    pub fn get_subsystem<T: SceneDataSubsystem>(&self) -> Option<&T> {
        self.subsystems.get::<T>()
    }

    /// Whether this opaque extension entity is still alive.
    pub fn is_alive(&self, entity: SceneExtensionEntity) -> bool {
        is_extension_entity(self.authority, entity)
    }

    /// Read one application component from an extension entity.
    pub fn get<T: SceneDataComponent>(&self, entity: SceneExtensionEntity) -> Option<&T> {
        is_extension_entity(self.authority, entity)
            .then(|| self.authority.get::<T>(entity.0))
            .flatten()
    }

    /// Iterate an application component without exposing raw SceneDB entity
    /// identities. The filter is normally free after type/domain discipline;
    /// it remains here as a defensive boundary check.
    pub fn query<T: SceneDataComponent>(
        &self,
    ) -> impl Iterator<Item = (SceneExtensionEntity, &T)> + '_ {
        self.authority.query::<T>().filter_map(move |(entity, value)| {
            self.authority
                .get::<ExtensionEntityMarker>(entity)
                .is_some()
                .then_some((SceneExtensionEntity(entity), value))
        })
    }

    /// Stable component-local GPU row for one application component.
    pub fn gpu_row<T: SceneDataComponent>(&self, entity: SceneExtensionEntity) -> Option<u32> {
        is_extension_entity(self.authority, entity)
            .then(|| self.authority.gpu_row::<T>(entity.0))
            .flatten()
    }

    /// Current addressable row span for an application component's GPU
    /// partner. The span may include reusable holes.
    pub fn gpu_row_span<T: SceneDataComponent>(&self) -> u32 {
        self.authority.gpu_row_span::<T>()
    }

    /// Number of live mirrored rows for an application component.
    pub fn gpu_live_count<T: SceneDataComponent>(&self) -> u32 {
        self.authority.gpu_live_count::<T>()
    }

    /// Snapshot one of `T`'s stable, explicitly named GPU partner buffers
    /// after the Scene has been flushed. Physical reallocations advance
    /// `epoch`.
    ///
    /// The cloned WGPU handle is published for read-only shader binding. WGPU
    /// cannot attenuate a buffer handle's usage flags, so writing through it
    /// would violate the facade contract. Requiring `T` and reflecting the key
    /// prevents extension code from obtaining Helio's built-in partner buffers.
    pub fn buffer<T: SceneDataComponent>(&self, key: &str) -> Option<SceneDataBuffer> {
        let descriptor = self
            .authority
            .gpu_column_descs_for_component(component_id::<T>())?
            .into_iter()
            .find(|column| column.buffer_key == Some(key))?;
        self.authority
            .gpu_store()
            .gpu_buffer_snapshot_for_id(descriptor.field_token.id())
            .map(|(buffer, epoch, descriptor)| SceneDataBuffer {
                buffer,
                epoch,
                row_stride: descriptor.value_token.desc().size,
            })
    }

    /// Typed publication by reflected field name. This is the fallback for a
    /// bare `#[gpu]` field or a packed component whose generated descriptor has
    /// no explicit stable buffer key. Prefer [`Self::buffer`] for renderer ABI
    /// contracts because an explicit key survives Rust field renames.
    pub fn field_buffer<T: SceneDataComponent>(&self, field: &str) -> Option<SceneDataBuffer> {
        let descriptor = self
            .authority
            .gpu_column_descs_for_component(component_id::<T>())?
            .into_iter()
            .find(|column| column.buffer_name == field)?;
        self.authority
            .gpu_store()
            .gpu_buffer_snapshot_for_id(descriptor.field_token.id())
            .map(|(buffer, epoch, descriptor)| SceneDataBuffer {
                buffer,
                epoch,
                row_stride: descriptor.value_token.desc().size,
            })
    }

    /// Snapshot the component-local presence buffer used to validate holes
    /// and removals in a custom GPU consumer.
    pub fn presence_buffer<T: SceneDataComponent>(&self) -> Option<SceneDataBuffer> {
        self.authority
            .gpu_store()
            .component_presence_buffer_snapshot_for_id(component_id::<T>())
            .map(|(buffer, epoch)| SceneDataBuffer {
                buffer,
                epoch,
                row_stride: std::mem::size_of::<u32>() as u32,
            })
    }
}

/// Mutable access to application-owned SceneDB data.
///
/// This facade owns no data. It borrows [`Scene`] so actor identity teardown
/// and SceneDB entity teardown stay atomic when an extension entity is
/// despawned.
pub struct SceneDataMut<'a> {
    scene: &'a mut Scene,
}

impl<'a> SceneDataMut<'a> {
    pub(super) fn new(scene: &'a mut Scene) -> Self {
        Self { scene }
    }

    pub fn view(&self) -> SceneDataView<'_> {
        SceneDataView::new(&self.scene.authority)
    }

    /// Read one application-owned global/index/query payload.
    pub fn get_subsystem<T: SceneDataSubsystem>(&self) -> Option<&T> {
        self.scene
            .authority
            .subsystem::<ExtensionSubsystemStore>()
            .expect("extension subsystem store is registered at Scene construction")
            .get::<T>()
    }

    /// Insert a new application-owned subsystem payload without replacing an
    /// existing value of the same concrete type.
    pub fn insert_subsystem<T: SceneDataSubsystem>(
        &mut self,
        value: T,
    ) -> Result<(), SceneDataError> {
        let inserted = self
            .scene
            .authority
            .subsystem_mut::<ExtensionSubsystemStore>()
            .expect("extension subsystem store is registered at Scene construction")
            .insert(value);
        if inserted {
            Ok(())
        } else {
            Err(SceneDataError::SubsystemAlreadyPresent {
                subsystem: type_name::<T>(),
            })
        }
    }

    /// Mutate an application-owned subsystem payload in its canonical SceneDB
    /// allocation.
    pub fn edit_subsystem<T: SceneDataSubsystem, R>(
        &mut self,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Result<R, SceneDataError> {
        self.scene
            .authority
            .subsystem_mut::<ExtensionSubsystemStore>()
            .expect("extension subsystem store is registered at Scene construction")
            .get_mut::<T>()
            .map(edit)
            .ok_or_else(subsystem_missing::<T>)
    }

    /// Remove and return one application-owned subsystem payload.
    pub fn remove_subsystem<T: SceneDataSubsystem>(&mut self) -> Result<T, SceneDataError> {
        self.scene
            .authority
            .subsystem_mut::<ExtensionSubsystemStore>()
            .expect("extension subsystem store is registered at Scene construction")
            .remove::<T>()
            .ok_or_else(subsystem_missing::<T>)
    }

    pub fn is_alive(&self, entity: SceneExtensionEntity) -> bool {
        is_extension_entity(&self.scene.authority, entity)
    }

    pub fn get<T: SceneDataComponent>(&self, entity: SceneExtensionEntity) -> Option<&T> {
        is_extension_entity(&self.scene.authority, entity)
            .then(|| self.scene.authority.get::<T>(entity.0))
            .flatten()
    }

    pub fn query<T: SceneDataComponent>(
        &self,
    ) -> impl Iterator<Item = (SceneExtensionEntity, &T)> + '_ {
        self.scene
            .authority
            .query::<T>()
            .filter_map(move |(entity, value)| {
                self.scene
                    .authority
                    .get::<ExtensionEntityMarker>(entity)
                    .is_some()
                    .then_some((SceneExtensionEntity(entity), value))
            })
    }

    pub fn gpu_row<T: SceneDataComponent>(&self, entity: SceneExtensionEntity) -> Option<u32> {
        self.view().gpu_row::<T>(entity)
    }

    pub fn gpu_row_span<T: SceneDataComponent>(&self) -> u32 {
        self.scene.authority.gpu_row_span::<T>()
    }

    pub fn gpu_live_count<T: SceneDataComponent>(&self) -> u32 {
        self.scene.authority.gpu_live_count::<T>()
    }

    pub fn buffer<T: SceneDataComponent>(&self, key: &str) -> Option<SceneDataBuffer> {
        self.view().buffer::<T>(key)
    }

    pub fn field_buffer<T: SceneDataComponent>(&self, field: &str) -> Option<SceneDataBuffer> {
        self.view().field_buffer::<T>(field)
    }

    pub fn presence_buffer<T: SceneDataComponent>(&self) -> Option<SceneDataBuffer> {
        self.view().presence_buffer::<T>()
    }

    /// Spawn an empty application-owned entity. Components can then be added
    /// independently without exposing raw SceneDB identity.
    pub fn spawn(&mut self) -> SceneExtensionEntity {
        self.scene.spawn_extension_entity()
    }

    /// Despawn an extension entity and detach any retained actor whose custom
    /// identity or target is that entity.
    pub fn despawn(&mut self, entity: SceneExtensionEntity) -> Result<(), SceneDataError> {
        if !is_extension_entity(&self.scene.authority, entity) {
            return Err(SceneDataError::InvalidEntity);
        }
        if !self
            .scene
            .detach_actors_for_target(SceneActorId::Custom(entity))
        {
            self.scene.authority.despawn(entity.0);
        }
        Ok(())
    }

    /// Spawn an extension entity containing one CPU-only component.
    pub fn insert_cpu<T: SceneCpuComponent>(
        &mut self,
        value: T,
    ) -> Result<SceneExtensionEntity, SceneDataError> {
        self.ensure_policy::<T>(ExtensionComponentPolicy::CpuOnly)?;
        let entity = self.scene.spawn_extension_entity();
        let inserted = self.scene.authority.insert_registered(entity.0, value);
        debug_assert!(inserted);
        Ok(entity)
    }

    /// Add a CPU-only component. Existing values must be changed through
    /// [`Self::edit_cpu`] so the operation stays in-place.
    pub fn add_cpu<T: SceneCpuComponent>(
        &mut self,
        entity: SceneExtensionEntity,
        value: T,
    ) -> Result<(), SceneDataError> {
        self.ensure_entity(entity)?;
        self.ensure_policy::<T>(ExtensionComponentPolicy::CpuOnly)?;
        self.ensure_absent::<T>(entity)?;
        let inserted = self.scene.authority.insert_registered(entity.0, value);
        debug_assert!(inserted);
        Ok(())
    }

    /// Mutate a CPU-only component in its canonical allocation.
    pub fn edit_cpu<T: SceneCpuComponent, R>(
        &mut self,
        entity: SceneExtensionEntity,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Result<R, SceneDataError> {
        self.ensure_entity(entity)?;
        self.ensure_policy::<T>(ExtensionComponentPolicy::CpuOnly)?;
        self.scene
            .authority
            .edit_registered_cpu::<T, R>(entity.0, edit)
            .ok_or_else(component_missing::<T>)
    }

    /// Spawn an extension entity containing one DirtyTracked GPU component.
    pub fn insert_gpu<T: SceneDirtyTrackedComponent>(
        &mut self,
        value: T,
    ) -> Result<SceneExtensionEntity, SceneDataError> {
        self.ensure_policy::<T>(ExtensionComponentPolicy::DirtyTracked)?;
        let entity = self.scene.spawn_extension_entity();
        let inserted = self.scene.authority.replace_gpu(entity.0, value);
        debug_assert!(inserted);
        Ok(entity)
    }

    /// Add a DirtyTracked GPU component when absent.
    pub fn add_gpu<T: SceneDirtyTrackedComponent>(
        &mut self,
        entity: SceneExtensionEntity,
        value: T,
    ) -> Result<(), SceneDataError> {
        self.ensure_entity(entity)?;
        self.ensure_policy::<T>(ExtensionComponentPolicy::DirtyTracked)?;
        self.ensure_absent::<T>(entity)?;
        let inserted = self.scene.authority.replace_gpu(entity.0, value);
        debug_assert!(inserted);
        Ok(())
    }

    /// Transactionally copy, edit, and reinsert a DirtyTracked component.
    /// SceneDB compares the old/new GPU fields and queues only real changes.
    pub fn edit_gpu<T: SceneDirtyTrackedComponent, R>(
        &mut self,
        entity: SceneExtensionEntity,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Result<R, SceneDataError> {
        self.ensure_entity(entity)?;
        self.ensure_policy::<T>(ExtensionComponentPolicy::DirtyTracked)?;
        self.scene
            .authority
            .edit_gpu::<T, R>(entity.0, edit)
            .ok_or_else(component_missing::<T>)
    }

    /// Spawn an extension entity containing one Once-handoff GPU component.
    pub fn insert_once<T: SceneOnceComponent>(
        &mut self,
        value: T,
    ) -> Result<SceneExtensionEntity, SceneDataError> {
        self.ensure_policy::<T>(ExtensionComponentPolicy::Once)?;
        let entity = self.scene.spawn_extension_entity();
        let inserted = self.scene.authority.insert_registered(entity.0, value);
        debug_assert!(inserted);
        Ok(entity)
    }

    /// Add a Once-handoff component only when it is absent. There is no edit
    /// or replacement API for this policy.
    pub fn add_once<T: SceneOnceComponent>(
        &mut self,
        entity: SceneExtensionEntity,
        value: T,
    ) -> Result<(), SceneDataError> {
        self.ensure_entity(entity)?;
        self.ensure_policy::<T>(ExtensionComponentPolicy::Once)?;
        self.ensure_absent::<T>(entity)?;
        // Ordinary World insertion plus policy validation and absence are what
        // preserve Once's one-time handoff.
        let inserted = self.scene.authority.insert_registered(entity.0, value);
        debug_assert!(inserted);
        Ok(())
    }

    /// Spawn an extension entity containing one mixed DirtyTracked/Once GPU
    /// component. Whole-component mutation is deliberately unavailable.
    pub fn insert_mixed<T: SceneMixedComponent>(
        &mut self,
        value: T,
    ) -> Result<SceneExtensionEntity, SceneDataError> {
        self.ensure_policy::<T>(ExtensionComponentPolicy::Mixed)?;
        let entity = self.scene.spawn_extension_entity();
        let inserted = self.scene.authority.insert_registered(entity.0, value);
        debug_assert!(inserted);
        Ok(entity)
    }

    /// Add a mixed DirtyTracked/Once component only while absent. Use
    /// [`Self::remove`] followed by this method for a new authored lifetime.
    pub fn add_mixed<T: SceneMixedComponent>(
        &mut self,
        entity: SceneExtensionEntity,
        value: T,
    ) -> Result<(), SceneDataError> {
        self.ensure_entity(entity)?;
        self.ensure_policy::<T>(ExtensionComponentPolicy::Mixed)?;
        self.ensure_absent::<T>(entity)?;
        let inserted = self.scene.authority.insert_registered(entity.0, value);
        debug_assert!(inserted);
        Ok(())
    }

    /// Remove any application component. Removing a Once component explicitly
    /// ends its handoff lifetime, so a later `add_once` is a fresh handoff.
    pub fn remove<T: SceneDataComponent>(
        &mut self,
        entity: SceneExtensionEntity,
    ) -> Result<T, SceneDataError> {
        self.ensure_entity(entity)?;
        self.scene
            .authority
            .remove::<T>(entity.0)
            .ok_or_else(component_missing::<T>)
    }

    fn ensure_entity(&self, entity: SceneExtensionEntity) -> Result<(), SceneDataError> {
        if is_extension_entity(&self.scene.authority, entity) {
            Ok(())
        } else {
            Err(SceneDataError::InvalidEntity)
        }
    }

    fn ensure_absent<T: SceneDataComponent>(
        &self,
        entity: SceneExtensionEntity,
    ) -> Result<(), SceneDataError> {
        if self.scene.authority.get::<T>(entity.0).is_none() {
            Ok(())
        } else {
            Err(SceneDataError::ComponentAlreadyPresent {
                component: type_name::<T>(),
            })
        }
    }

    fn ensure_policy<T: SceneDataComponent>(
        &mut self,
        expected: ExtensionComponentPolicy,
    ) -> Result<(), SceneDataError> {
        let type_id = TypeId::of::<T>();
        if let Some(&actual) = self.scene.extension_component_policies.get(&type_id) {
            return policy_matches::<T>(expected, actual);
        }

        let descriptors = self
            .scene
            .authority
            .gpu_column_descs_for_component(component_id::<T>());
        let actual = match descriptors.as_deref() {
            None | Some([]) => ExtensionComponentPolicy::CpuOnly,
            Some(columns) if columns.iter().all(|column| column.mode == MirrorMode::DirtyTracked) => {
                ExtensionComponentPolicy::DirtyTracked
            }
            Some(columns) if columns.iter().all(|column| column.mode == MirrorMode::Once) => {
                ExtensionComponentPolicy::Once
            }
            Some(_) => ExtensionComponentPolicy::Mixed,
        };
        policy_matches::<T>(expected, actual)?;

        if actual != ExtensionComponentPolicy::CpuOnly {
            let store = self.scene.authority.gpu_store();
            let registered = store
                .component_presence_buffer_snapshot_for_id(component_id::<T>())
                .is_some()
                && descriptors.as_ref().is_some_and(|columns| {
                    columns.iter().all(|column| {
                        store
                            .gpu_buffer_snapshot_for_id(column.field_token.id())
                            .is_some()
                    })
                });
            if !registered {
                return Err(SceneDataError::GpuColumnsNotRegistered {
                    component: type_name::<T>(),
                });
            }
        }

        self.scene
            .extension_component_policies
            .insert(type_id, actual);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::scene) enum ExtensionComponentPolicy {
    CpuOnly,
    DirtyTracked,
    Once,
    Mixed,
}

impl ExtensionComponentPolicy {
    fn label(self) -> &'static str {
        match self {
            Self::CpuOnly => "CPU-only",
            Self::DirtyTracked => "DirtyTracked",
            Self::Once => "Once",
            Self::Mixed => "mixed DirtyTracked/Once",
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::scene) struct ExtensionEntityMarker;

impl Scene {
    /// Read application-owned SceneDB components without exposing the
    /// authority or World that also contain Helio's built-ins.
    pub fn scene_data(&self) -> SceneDataView<'_> {
        SceneDataView::new(&self.authority)
    }

    /// Mutate application-owned SceneDB components through their declared
    /// CPU, DirtyTracked, or Once policy.
    pub fn scene_data_mut(&mut self) -> SceneDataMut<'_> {
        SceneDataMut::new(self)
    }

    pub(in crate::scene) fn spawn_extension_entity(&mut self) -> SceneExtensionEntity {
        SceneExtensionEntity(self.authority.insert(ExtensionEntityMarker))
    }

    pub(in crate::scene) fn clear_extension_data(&mut self) {
        let entities: Vec<_> = self
            .authority
            .query::<ExtensionEntityMarker>()
            .map(|(entity, _)| entity)
            .collect();
        for entity in entities {
            self.authority.despawn(entity);
        }
        self.authority
            .subsystem_mut::<ExtensionSubsystemStore>()
            .expect("extension subsystem store is registered at Scene construction")
            .clear();
    }
}

fn is_extension_entity(authority: &SceneAuthority, entity: SceneExtensionEntity) -> bool {
    authority.get::<ExtensionEntityMarker>(entity.0).is_some()
}

fn component_missing<T>() -> SceneDataError {
    SceneDataError::ComponentMissing {
        component: type_name::<T>(),
    }
}

fn subsystem_missing<T>() -> SceneDataError {
    SceneDataError::SubsystemMissing {
        subsystem: type_name::<T>(),
    }
}

fn policy_matches<T>(
    expected: ExtensionComponentPolicy,
    actual: ExtensionComponentPolicy,
) -> Result<(), SceneDataError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SceneDataError::ComponentPolicyMismatch {
            component: type_name::<T>(),
            expected: expected.label(),
            actual: actual.label(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pulsar_scenedb_derive::SceneStore;

    use super::*;

    const DIRTY_BUFFER_KEY: &str = "helio.test.extension.dirty";
    const ONCE_BUFFER_KEY: &str = "helio.test.extension.once";
    const MIXED_DIRTY_BUFFER_KEY: &str = "helio.test.extension.mixed.dirty";
    const MIXED_ONCE_BUFFER_KEY: &str = "helio.test.extension.mixed.once";

    #[derive(Debug, PartialEq, Eq)]
    struct CpuPayload {
        value: String,
    }

    impl SceneDataComponent for CpuPayload {}
    impl SceneCpuComponent for CpuPayload {}

    #[derive(Debug, PartialEq, Eq)]
    struct SubsystemPayload {
        value: String,
    }

    impl SceneDataSubsystem for SubsystemPayload {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq, SceneStore)]
    struct DirtyPayload {
        #[gpu(buffer = "helio.test.extension.dirty")]
        value: u32,
        cpu_tag: u32,
    }

    impl SceneDataComponent for DirtyPayload {}
    impl SceneDirtyTrackedStorage for DirtyPayload {}
    // Deliberately dishonest opt-in used to prove runtime reflection closes
    // the overlap possible between downstream-implementable marker traits.
    impl SceneCpuComponent for DirtyPayload {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq, SceneStore)]
    struct OncePayload {
        #[gpu(mirror = Once, buffer = "helio.test.extension.once")]
        value: u32,
    }

    impl SceneDataComponent for OncePayload {}
    impl SceneOnceComponent for OncePayload {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq, SceneStore)]
    struct MixedPayload {
        #[gpu(buffer = "helio.test.extension.mixed.dirty")]
        dirty: u32,
        #[gpu(mirror = Once, buffer = "helio.test.extension.mixed.once")]
        once: u32,
    }

    impl SceneDataComponent for MixedPayload {}
    impl SceneMixedComponent for MixedPayload {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq, SceneStore)]
    struct UnregisteredBuiltInKeySpoof {
        #[gpu(buffer = "helio.scene.lights")]
        value: u32,
    }

    impl SceneDataComponent for UnregisteredBuiltInKeySpoof {}

    fn test_scene() -> (Scene, Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let (device, queue) = crate::test_support::test_gpu().expect("No test GPU adapter found");
        let scene = Scene::new_with_component_registration(
            Arc::clone(&device),
            Arc::clone(&queue),
            |registrar| {
                registrar.register::<DirtyPayload>(1);
                registrar.register::<OncePayload>(1);
                registrar.register::<MixedPayload>(1);
            },
        );
        (scene, device, queue)
    }

    fn read_u32(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        offset: u64,
    ) -> u32 {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("helio-extension-test-readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("helio-extension-test-readback"),
        });
        encoder.copy_buffer_to_buffer(source, offset, &staging, 0, 4);
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("map readback"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll readback");
        let mapped = slice.get_mapped_range().expect("mapped readback");
        let value = u32::from_ne_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        value
    }

    #[test]
    fn cpu_extension_crud_is_in_place_and_domain_checked() {
        let (mut scene, _, _) = test_scene();
        let entity = scene
            .scene_data_mut()
            .insert_cpu(CpuPayload {
                value: "before".into(),
            })
            .unwrap();
        let before = scene.scene_data().get::<CpuPayload>(entity).unwrap() as *const CpuPayload;

        scene
            .scene_data_mut()
            .edit_cpu::<CpuPayload, _>(entity, |payload| payload.value = "after".into())
            .unwrap();
        let after = scene.scene_data().get::<CpuPayload>(entity).unwrap() as *const CpuPayload;
        assert_eq!(before, after, "CPU-only edits must not migrate or replace the component");
        assert_eq!(
            scene.scene_data().get::<CpuPayload>(entity).unwrap().value,
            "after"
        );
        assert_eq!(scene.scene_data().query::<CpuPayload>().count(), 1);

        let removed = scene.scene_data_mut().remove::<CpuPayload>(entity).unwrap();
        assert_eq!(removed.value, "after");
        assert!(matches!(
            scene.scene_data_mut().remove::<CpuPayload>(entity),
            Err(SceneDataError::ComponentMissing { .. })
        ));
        scene.scene_data_mut().despawn(entity).unwrap();
        assert!(!scene.scene_data().is_alive(entity));
        assert_eq!(
            scene.scene_data_mut().despawn(entity),
            Err(SceneDataError::InvalidEntity)
        );
    }

    #[test]
    fn subsystem_payload_crud_rejects_duplicates_and_reports_missing_types() {
        let (mut scene, _, _) = test_scene();
        assert!(scene
            .scene_data()
            .get_subsystem::<SubsystemPayload>()
            .is_none());
        assert!(matches!(
            scene
                .scene_data_mut()
                .edit_subsystem::<SubsystemPayload, _>(|payload| payload.value.clear()),
            Err(SceneDataError::SubsystemMissing { .. })
        ));
        assert!(matches!(
            scene
                .scene_data_mut()
                .remove_subsystem::<SubsystemPayload>(),
            Err(SceneDataError::SubsystemMissing { .. })
        ));

        scene
            .scene_data_mut()
            .insert_subsystem(SubsystemPayload {
                value: "first".into(),
            })
            .unwrap();
        assert!(matches!(
            scene
                .scene_data_mut()
                .insert_subsystem(SubsystemPayload {
                    value: "duplicate".into(),
                }),
            Err(SceneDataError::SubsystemAlreadyPresent { .. })
        ));
        assert_eq!(
            scene
                .scene_data()
                .get_subsystem::<SubsystemPayload>()
                .unwrap()
                .value,
            "first"
        );

        let old = scene
            .scene_data_mut()
            .edit_subsystem::<SubsystemPayload, _>(|payload| {
                std::mem::replace(&mut payload.value, "edited".into())
            })
            .unwrap();
        assert_eq!(old, "first");
        let removed = scene
            .scene_data_mut()
            .remove_subsystem::<SubsystemPayload>()
            .unwrap();
        assert_eq!(removed.value, "edited");
        assert!(scene
            .scene_data()
            .get_subsystem::<SubsystemPayload>()
            .is_none());
    }

    #[test]
    fn dirty_gpu_edits_reinsert_and_publish_named_buffer_and_presence() {
        let (mut scene, device, queue) = test_scene();
        let entity = scene
            .scene_data_mut()
            .insert_gpu(DirtyPayload {
                value: 7,
                cpu_tag: 1,
            })
            .unwrap();
        let row = scene.scene_data().gpu_row::<DirtyPayload>(entity).unwrap();
        scene.flush();

        let first = scene
            .scene_data()
            .buffer::<DirtyPayload>(DIRTY_BUFFER_KEY)
            .unwrap();
        assert!(scene
            .scene_data()
            .buffer::<DirtyPayload>("helio.scene.lights")
            .is_none());
        assert!(scene
            .scene_data()
            .buffer::<DirtyPayload>(ONCE_BUFFER_KEY)
            .is_none());
        let by_field = scene
            .scene_data()
            .field_buffer::<DirtyPayload>("value")
            .unwrap();
        assert_eq!(by_field.row_stride, first.row_stride);
        assert!(scene
            .scene_data()
            .field_buffer::<DirtyPayload>("not_a_field")
            .is_none());
        assert_eq!(first.row_stride, 4);
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &first.buffer,
                u64::from(row) * u64::from(first.row_stride),
            ),
            7
        );
        let presence = scene
            .scene_data()
            .presence_buffer::<DirtyPayload>()
            .unwrap();
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &presence.buffer,
                u64::from(row) * u64::from(presence.row_stride),
            ),
            1
        );

        let old_cpu_tag = scene
            .scene_data_mut()
            .edit_gpu::<DirtyPayload, _>(entity, |payload| {
                let old = payload.cpu_tag;
                payload.value = 91;
                payload.cpu_tag = 2;
                old
            })
            .unwrap();
        assert_eq!(old_cpu_tag, 1);
        scene.flush();
        let second = scene
            .scene_data()
            .buffer::<DirtyPayload>(DIRTY_BUFFER_KEY)
            .unwrap();
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &second.buffer,
                u64::from(row) * u64::from(second.row_stride),
            ),
            91
        );

        let spare = scene.scene_data_mut().spawn();
        assert!(matches!(
            scene.scene_data_mut().add_cpu(
                spare,
                DirtyPayload {
                    value: 1,
                    cpu_tag: 0,
                },
            ),
            Err(SceneDataError::ComponentPolicyMismatch { .. })
        ));
    }

    #[test]
    fn unregistered_matching_key_cannot_publish_a_builtin_partner() {
        let (scene, _, _) = test_scene();
        assert!(scene
            .scene_data()
            .buffer::<UnregisteredBuiltInKeySpoof>("helio.scene.lights")
            .is_none());
        assert!(scene
            .scene_data()
            .field_buffer::<UnregisteredBuiltInKeySpoof>("value")
            .is_none());
        assert!(scene
            .scene_data()
            .presence_buffer::<UnregisteredBuiltInKeySpoof>()
            .is_none());
    }

    #[test]
    fn once_components_require_absence_and_readd_starts_a_new_handoff() {
        let (mut scene, device, queue) = test_scene();
        let entity = scene.scene_data_mut().spawn();
        scene
            .scene_data_mut()
            .add_once(entity, OncePayload { value: 11 })
            .unwrap();
        assert!(matches!(
            scene
                .scene_data_mut()
                .add_once(entity, OncePayload { value: 12 }),
            Err(SceneDataError::ComponentAlreadyPresent { .. })
        ));
        scene.flush();
        let first_row = scene.scene_data().gpu_row::<OncePayload>(entity).unwrap();
        let first = scene
            .scene_data()
            .buffer::<OncePayload>(ONCE_BUFFER_KEY)
            .unwrap();
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &first.buffer,
                u64::from(first_row) * u64::from(first.row_stride),
            ),
            11
        );

        scene.scene_data_mut().remove::<OncePayload>(entity).unwrap();
        scene
            .scene_data_mut()
            .add_once(entity, OncePayload { value: 42 })
            .unwrap();
        scene.flush();
        let second_row = scene.scene_data().gpu_row::<OncePayload>(entity).unwrap();
        let second = scene
            .scene_data()
            .buffer::<OncePayload>(ONCE_BUFFER_KEY)
            .unwrap();
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &second.buffer,
                u64::from(second_row) * u64::from(second.row_stride),
            ),
            42
        );
    }

    #[test]
    fn mixed_components_are_lifecycle_only_and_readd_begins_both_lifetimes() {
        let (mut scene, device, queue) = test_scene();
        let entity = scene.scene_data_mut().spawn();
        scene
            .scene_data_mut()
            .add_mixed(
                entity,
                MixedPayload {
                    dirty: 3,
                    once: 4,
                },
            )
            .unwrap();
        assert!(matches!(
            scene.scene_data_mut().add_mixed(
                entity,
                MixedPayload {
                    dirty: 30,
                    once: 40,
                },
            ),
            Err(SceneDataError::ComponentAlreadyPresent { .. })
        ));
        scene.flush();
        let first_row = scene.scene_data().gpu_row::<MixedPayload>(entity).unwrap();
        let dirty = scene
            .scene_data()
            .buffer::<MixedPayload>(MIXED_DIRTY_BUFFER_KEY)
            .unwrap();
        let once = scene
            .scene_data()
            .buffer::<MixedPayload>(MIXED_ONCE_BUFFER_KEY)
            .unwrap();
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &dirty.buffer,
                u64::from(first_row) * u64::from(dirty.row_stride),
            ),
            3
        );
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &once.buffer,
                u64::from(first_row) * u64::from(once.row_stride),
            ),
            4
        );

        scene.scene_data_mut().remove::<MixedPayload>(entity).unwrap();
        scene
            .scene_data_mut()
            .add_mixed(
                entity,
                MixedPayload {
                    dirty: 31,
                    once: 41,
                },
            )
            .unwrap();
        scene.flush();
        let second_row = scene.scene_data().gpu_row::<MixedPayload>(entity).unwrap();
        let dirty = scene
            .scene_data()
            .buffer::<MixedPayload>(MIXED_DIRTY_BUFFER_KEY)
            .unwrap();
        let once = scene
            .scene_data()
            .buffer::<MixedPayload>(MIXED_ONCE_BUFFER_KEY)
            .unwrap();
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &dirty.buffer,
                u64::from(second_row) * u64::from(dirty.row_stride),
            ),
            31
        );
        assert_eq!(
            read_u32(
                &device,
                &queue,
                &once.buffer,
                u64::from(second_row) * u64::from(once.row_stride),
            ),
            41
        );
    }

    #[test]
    fn scene_clear_despawns_application_entities() {
        let (mut scene, _, _) = test_scene();
        let entity = scene
            .scene_data_mut()
            .insert_cpu(CpuPayload {
                value: "owned".into(),
            })
            .unwrap();
        scene
            .scene_data_mut()
            .insert_subsystem(SubsystemPayload {
                value: "global".into(),
            })
            .unwrap();
        scene.clear();
        assert!(!scene.scene_data().is_alive(entity));
        assert_eq!(scene.scene_data().query::<CpuPayload>().count(), 0);
        assert!(scene
            .scene_data()
            .get_subsystem::<SubsystemPayload>()
            .is_none());
        scene
            .scene_data_mut()
            .insert_subsystem(SubsystemPayload {
                value: "new-scene".into(),
            })
            .expect("clear keeps the private SceneDB container registered");
    }
}
