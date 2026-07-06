use crate::traits::{MaybeSend, MaybeSync};
use std::any::{Any, TypeId};

pub trait ComponentVec: 'static {
    fn type_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn len(&self) -> usize;
    fn clear(&mut self);
    fn is_empty(&self) -> bool;
}

impl<T: Component> ComponentVec for Vec<T> {
    fn type_name(&self) -> &'static str {
        T::type_name()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn clear(&mut self) {
        self.clear()
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

pub trait Component: MaybeSend + MaybeSync + 'static {
    fn type_name() -> &'static str
    where
        Self: Sized,
    {
        std::any::type_name::<Self>()
    }
}

pub struct ComponentRegistry {
    storage: Vec<Box<dyn ComponentVec>>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            storage: Vec::new(),
        }
    }

    pub fn register<T: Component + 'static>(&mut self) {
        self.storage.push(Box::new(Vec::<T>::new()));
    }

    pub fn get_storage<T: Component + 'static>(&self) -> Option<&Vec<T>> {
        let tid = TypeId::of::<T>();
        self.storage
            .iter()
            .find(|v| v.as_any().type_id() == tid)
            .and_then(|v| v.as_any().downcast_ref::<Vec<T>>())
    }

    pub fn get_storage_mut<T: Component + 'static>(&mut self) -> Option<&mut Vec<T>> {
        let tid = TypeId::of::<T>();
        self.storage
            .iter_mut()
            .find(|v| v.as_any().type_id() == tid)
            .and_then(|v| v.as_any_mut().downcast_mut::<Vec<T>>())
    }

    pub fn register_with<T: Component + 'static, F: FnOnce(&mut Vec<T>)>(&mut self, f: F) {
        let mut vec = Vec::<T>::new();
        f(&mut vec);
        self.storage.push(Box::new(vec));
    }

    pub fn storage_count(&self) -> usize {
        self.storage.len()
    }
}

pub struct ComponentSlot<T> {
    pub index: u32,
    pub generation: u32,
    pub _phantom: std::marker::PhantomData<T>,
}
