use crate::{
    Buffer, BufferDescriptor, BufferUsages, Device, Extent3d, Queue, Texture, TextureDescriptor,
};

/// Data used to initialize a buffer at creation time.
#[derive(Clone, Debug)]
pub struct BufferInitDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub contents: &'a [u8],
    pub usage: BufferUsages,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum TextureDataOrder {
    #[default]
    LayerMajor,
    MipMajor,
}

/// Convenience methods retained for compatibility with Helio pass crates.
pub trait DeviceExt {
    fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Buffer;

    fn create_texture_with_data(
        &self,
        queue: &Queue,
        desc: &TextureDescriptor<'_>,
        order: TextureDataOrder,
        data: &[u8],
    ) -> Texture;
}

impl DeviceExt for Device {
    fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Buffer {
        let size = desc.contents.len().max(4) as u64;
        let buffer = self.create_buffer(&BufferDescriptor {
            label: desc.label,
            size,
            usage: desc.usage | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue().write_buffer(&buffer, 0, desc.contents);
        buffer
    }

    fn create_texture_with_data(
        &self,
        queue: &Queue,
        desc: &TextureDescriptor<'_>,
        _order: TextureDataOrder,
        data: &[u8],
    ) -> Texture {
        let texture = self.create_texture(desc);
        let block_size = desc.format.block_copy_size(None).unwrap_or(4);
        queue.write_texture(
            crate::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: crate::Origin3d::ZERO,
                aspect: crate::TextureAspect::All,
            },
            data,
            crate::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(desc.size.width * block_size),
                rows_per_image: Some(desc.size.height),
            },
            Extent3d {
                width: desc.size.width,
                height: desc.size.height,
                depth_or_array_layers: desc.size.depth_or_array_layers,
            },
        );
        texture
    }
}
