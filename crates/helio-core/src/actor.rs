use crate::component::Component;
use crate::entity::Entity;

pub trait Actor: Component {
    fn entity(&self) -> Entity;
    fn set_entity(&mut self, entity: Entity);
}
