# Helio → TRUEOS build artifact

The first supported program is intentionally only Helio's `SimpleCubePass` in
`helio_default_graphs::build_simple_graph`. Its purpose is to establish a
repeatable frontend boundary, not to special-case enough Intel commands to put
one cube on screen.

Build it with:

```sh
just bake-trueos-cube
```

This executes one headless frame through `Renderer::render`, the real render
graph, and the vendored wgpu 30 native backend. wgpu validation records the
shader, buffer uploads, bind groups, pipeline state, render attachments, and
the indexed draw into its API trace. The baker verifies that the trace contains
the `SimpleCube` pipeline, 864-byte vertex buffer, 72-byte index buffer, WGSL
entry points, and one 36-index draw before producing
`target/helio-artifacts/simple-cube.trueos.helio`.

The baker does not contain cube vertices, indices, WGSL, or a second pipeline
description. Those enter the artifact only because `SimpleCubePass` created
and used them through wgpu.

## Artifact boundary

`helio-artifact` owns the little-endian `HELIOA` container. Every section is
named, typed, bounds-checked, and CRC32-protected. Version 1 contains:

- `manifest.json`, including the graph identity, target, output format, and
  dynamic `camera.view_proj` and `output.surface` slots;
- the complete wgpu trace under `wgpu/`;
- capture adapter metadata, kept separate from target metadata.

The wgpu trace's pointer-shaped IDs are symbolic object identities. They are
not GPU virtual addresses and must never be copied into a TRUEOS batch. The
lowerer should canonicalize these IDs, follow references from the submitted
render pass, and discard unreferenced renderer setup resources.

The container already assigns section kinds for Intel Xe-LP ISA and compiler
metadata. They are deliberately absent today: a trace-only artifact is a
frontend artifact, not yet a TRUEOS-executable game.

## Paved next increments

1. Add a trace lowerer that emits a normalized resource/pipeline/pass IR and
   proves the single submitted pass is `SimpleCube` without label matching.
2. Compile the captured WGSL vertex and fragment entry points for the TRUEOS
   Intel Xe-LP target and append ISA plus compiler metadata sections.
3. Add the TRUEOS `HELIOA` loader: validate sections, allocate resources, patch
   symbolic resource references, and encode fresh 3DSTATE/MI commands.
4. Submit that batch through `KernelClient::Render`, then publish its BGRA8
   surface through UI4.
5. Replace the simple graph with a second Helio graph without adding any
   scene-specific code to TRUEOS. That is the test that this became a game
   path rather than a cube stunt.

The loader should reject trace-only artifacts until steps 1–2 have supplied
the normalized IR and native shader sections. This keeps a clear distinction
between “captured by the real engine” and “executable by TRUEOS.”
