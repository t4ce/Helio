//! SceneDB-owned compiled Radiant material-graph assets.
//!
//! Graph WGSL is persistent authored/asset data, not render-pass state. The
//! registry therefore lives beside the rest of SceneDB's stateful subsystems;
//! Helio publishes a cloned lookup projection only when
//! [`RadiantGraphRegistry::epoch`] changes. A scene-content clear deliberately
//! keeps this reusable asset registry alive, just like it keeps non-recycling
//! asset-key allocator history.

use std::collections::HashMap;

use pulsar_scenedb::Subsystem;

/// Canonical Radiant graph asset registration failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiantGraphError {
    /// Material graph hash zero is the renderer ABI's "no graph" sentinel.
    ReservedHash,
}

impl std::fmt::Display for RadiantGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReservedHash => write!(f, "Radiant graph hash zero is reserved"),
        }
    }
}

impl std::error::Error for RadiantGraphError {}

/// Content-addressed registry of graph-generated WGSL snippets.
///
/// `graph_hash` is supplied by the external graph compiler. Re-registering an
/// identical `(hash, source)` pair is allocation-free and does not advance the
/// content epoch. Replacing source under the same hash is supported for editor
/// hot reload and does advance it, so every renderer pipeline cache can retire
/// the old compiled source deterministically.
pub struct RadiantGraphRegistry {
    snippets: HashMap<u64, String>,
    epoch: u64,
}

impl RadiantGraphRegistry {
    pub fn new() -> Self {
        Self {
            snippets: HashMap::new(),
            epoch: 0,
        }
    }

    /// Register or hot-replace a compiled graph snippet transactionally.
    /// Hash zero is rejected before either content or epoch can change.
    pub fn register(
        &mut self,
        graph_hash: u64,
        wgsl_snippet: String,
    ) -> Result<(), RadiantGraphError> {
        if graph_hash == 0 {
            return Err(RadiantGraphError::ReservedHash);
        }
        if self.snippets.get(&graph_hash).map(String::as_str) == Some(wgsl_snippet.as_str()) {
            return Ok(());
        }
        self.snippets.insert(graph_hash, wgsl_snippet);
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    /// Look up the canonical compiled source for a graph hash.
    pub fn get(&self, graph_hash: u64) -> Option<&str> {
        self.snippets.get(&graph_hash).map(String::as_str)
    }

    /// Remove a compiled graph asset. Missing hashes are an idempotent no-op.
    pub fn unregister(&mut self, graph_hash: u64) {
        if self.snippets.remove(&graph_hash).is_some() {
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    pub fn len(&self) -> usize {
        self.snippets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snippets.is_empty()
    }

    /// Changes only when canonical registry content changes.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Clone the canonical sources into Helio's render-facing lookup cache.
    /// Callers should gate this O(total WGSL bytes) operation on [`Self::epoch`].
    pub fn snapshot(&self) -> HashMap<u64, String> {
        self.snippets.clone()
    }
}

impl Default for RadiantGraphRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Subsystem for RadiantGraphRegistry {
    fn name(&self) -> &'static str {
        "helio.scene.radiant_graph_registry"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{RadiantGraphError, RadiantGraphRegistry};

    #[test]
    fn content_epoch_covers_same_hash_replacement_and_removal() {
        let mut registry = RadiantGraphRegistry::new();
        registry.register(7, "first".to_owned()).unwrap();
        let inserted = registry.epoch();

        registry.register(7, "first".to_owned()).unwrap();
        assert_eq!(registry.epoch(), inserted, "byte-identical updates are clean");

        registry.register(7, "replacement".to_owned()).unwrap();
        let replaced = registry.epoch();
        assert_eq!(replaced, inserted.wrapping_add(1));
        assert_eq!(registry.get(7), Some("replacement"));

        registry.unregister(99);
        assert_eq!(registry.epoch(), replaced, "missing removal is clean");

        registry.unregister(7);
        assert_eq!(registry.epoch(), replaced.wrapping_add(1));
        assert_eq!(registry.get(7), None);
        assert!(registry.is_empty());
    }

    #[test]
    fn reserved_zero_hash_is_rejected_without_mutation() {
        let mut registry = RadiantGraphRegistry::new();
        assert_eq!(
            registry.register(0, "unreachable".to_owned()),
            Err(RadiantGraphError::ReservedHash)
        );
        assert_eq!(registry.epoch(), 0);
        assert!(registry.is_empty());
        assert_eq!(registry.get(0), None);
    }
}
