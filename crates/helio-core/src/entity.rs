#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(u64);

impl Entity {
    pub fn index(self) -> u32 {
        self.0 as u32
    }
    pub fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    pub fn to_raw(self) -> u64 {
        self.0
    }
    pub fn null() -> Self {
        Self(u64::MAX)
    }
    pub fn is_null(self) -> bool {
        self.0 == u64::MAX
    }
}
