use crate::constants::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Empty,
    Solid(u8),
    Leaf,
    Branch,
}

#[derive(Debug, Clone)]
pub struct OctreeNode {
    pub state: NodeState,
    pub gpu_slot: u32,
    pub dirty: bool,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub children: Option<Box<[Option<Box<OctreeNode>>; 8]>>,
}

impl OctreeNode {
    pub fn new_empty(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            state: NodeState::Empty,
            gpu_slot: BRICK_EMPTY,
            dirty: false,
            aabb_min,
            aabb_max,
            children: None,
        }
    }

    pub fn new_solid(material: u8, aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            state: NodeState::Solid(material),
            gpu_slot: BRICK_SOLID,
            dirty: false,
            aabb_min,
            aabb_max,
            children: None,
        }
    }

    pub fn new_leaf(gpu_slot: u32, aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            state: NodeState::Leaf,
            gpu_slot,
            dirty: true,
            aabb_min,
            aabb_max,
            children: None,
        }
    }

    pub fn subdivide(&mut self) {
        if self.children.is_some() {
            return;
        }
        let mid = [
            (self.aabb_min[0] + self.aabb_max[0]) * 0.5,
            (self.aabb_min[1] + self.aabb_max[1]) * 0.5,
            (self.aabb_min[2] + self.aabb_max[2]) * 0.5,
        ];
        let mut children: [Option<Box<OctreeNode>>; 8] = Default::default();
        for i in 0..8 {
            let (lx, ly, lz) = ((i & 1) != 0, (i & 2) != 0, (i & 4) != 0);
            let min = [
                if lx { mid[0] } else { self.aabb_min[0] },
                if ly { mid[1] } else { self.aabb_min[1] },
                if lz { mid[2] } else { self.aabb_min[2] },
            ];
            let max = [
                if lx { self.aabb_max[0] } else { mid[0] },
                if ly { self.aabb_max[1] } else { mid[1] },
                if lz { self.aabb_max[2] } else { mid[2] },
            ];
            children[i] = Some(Box::new(OctreeNode::new_empty(min, max)));
        }
        self.children = Some(Box::new(children));
        self.state = NodeState::Branch;
    }

    pub fn collect_leaves<'a>(&'a self, leaves: &mut Vec<&'a OctreeNode>) {
        match self.state {
            NodeState::Leaf => leaves.push(self),
            NodeState::Branch => {
                if let Some(ref children) = self.children {
                    for child in children.iter().flatten() {
                        child.collect_leaves(leaves);
                    }
                }
            }
            _ => {}
        }
    }

    pub fn collect_dirty_leaves<'a>(&'a self, leaves: &mut Vec<&'a OctreeNode>) {
        if self.dirty {
            if matches!(self.state, NodeState::Leaf) {
                leaves.push(self);
            }
        }
        if let Some(ref children) = self.children {
            for child in children.iter().flatten() {
                child.collect_dirty_leaves(leaves);
            }
        }
    }

    pub fn mark_sphere_dirty(&mut self, center: [f32; 3], radius: f32) -> bool {
        let cx = (self.aabb_min[0] + self.aabb_max[0]) * 0.5;
        let cy = (self.aabb_min[1] + self.aabb_max[1]) * 0.5;
        let cz = (self.aabb_min[2] + self.aabb_max[2]) * 0.5;
        let hx = (self.aabb_max[0] - self.aabb_min[0]) * 0.5;
        let hy = (self.aabb_max[1] - self.aabb_min[1]) * 0.5;
        let hz = (self.aabb_max[2] - self.aabb_min[2]) * 0.5;
        let dx = (center[0] - cx).abs();
        let dy = (center[1] - cy).abs();
        let dz = (center[2] - cz).abs();
        if dx > hx + radius || dy > hy + radius || dz > hz + radius {
            return false;
        }
        match &mut self.state {
            NodeState::Leaf => {
                self.dirty = true;
                true
            }
            NodeState::Branch => {
                if let Some(ref mut children) = self.children {
                    let mut any = false;
                    for child in children.iter_mut().flatten() {
                        any |= child.mark_sphere_dirty(center, radius);
                    }
                    if any { self.dirty = true; }
                    any
                } else { false }
            }
            NodeState::Empty | NodeState::Solid(_) => false,
        }
    }
}

#[derive(Debug)]
pub struct VoxelOctree {
    pub root: OctreeNode,
    pub brick_size: u32,
    pub voxel_size: f32,
    pub level_count: u32,
}

impl VoxelOctree {
    pub fn new(voxel_size: f32, root_extent: f32) -> Self {
        let half = root_extent * 0.5;
        Self {
            root: OctreeNode::new_empty([-half, -half, -half], [half, half, half]),
            brick_size: BRICK_SIZE,
            voxel_size,
            level_count: 1,
        }
    }

    pub fn needed_depth(world_size: f32, voxel_size: f32, brick_size: u32) -> u32 {
        let brick_world = brick_size as f32 * voxel_size;
        let mut depth = 0;
        let mut size = world_size;
        while size > brick_world {
            size *= 0.5;
            depth += 1;
        }
        depth
    }

    pub fn ensure_depth(&mut self, target_depth: u32) {
        while self.level_count < target_depth {
            self.subdivide_root();
        }
    }

    fn subdivide_root(&mut self) {
        if !matches!(self.root.state, NodeState::Branch) {
            self.root.subdivide();
            self.level_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octree_basic() {
        let mut octree = VoxelOctree::new(1.0, 256.0);
        assert_eq!(octree.level_count, 1);
        assert!(matches!(octree.root.state, NodeState::Empty));
    }

    #[test]
    fn test_subdivide() {
        let mut octree = VoxelOctree::new(1.0, 256.0);
        octree.ensure_depth(4);
        assert!(octree.level_count >= 4);
        assert!(matches!(octree.root.state, NodeState::Branch));
    }

    #[test]
    fn test_mark_dirty() {
        let mut octree = VoxelOctree::new(1.0, 256.0);
        octree.root.subdivide();
        if let Some(ref mut children) = octree.root.children {
            children[0] = Some(Box::new(OctreeNode::new_leaf(0, [0.0; 3], [1.0; 3])));
        }
        assert!(octree.root.mark_sphere_dirty([0.5, 0.5, 0.5], 2.0));
    }

    #[test]
    fn test_collect_leaves() {
        let mut octree = VoxelOctree::new(1.0, 256.0);
        octree.ensure_depth(3);
        let mut leaves = vec![];
        octree.root.collect_leaves(&mut leaves);
        assert_eq!(leaves.len(), 512);
    }
}
