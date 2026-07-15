//! A concrete Rust wrapper over the browser's WebGPU JavaScript API.
//!
//! This crate deliberately implements only the WebGPU surface Helio uses. It
//! has no native backends, backend dispatch, shader translation, or GPU-side
//! validation layer; those responsibilities belong to the browser.

#![allow(clippy::too_many_arguments)]

mod api;
mod js;
mod types;
pub mod util;

pub use api::*;
pub use types::*;

pub use raw_window_handle as rwh;
pub use web_sys;

pub type PollType = api::PollType<SubmissionIndex>;

/// Raw browser WebGPU handles for interoperability.
pub mod webgpu {
    pub type GpuBuffer = wasm_bindgen::JsValue;
    pub type GpuDevice = wasm_bindgen::JsValue;
    pub type GpuQueue = wasm_bindgen::JsValue;
    pub type GpuTexture = wasm_bindgen::JsValue;
    pub type GpuTextureView = wasm_bindgen::JsValue;
}
