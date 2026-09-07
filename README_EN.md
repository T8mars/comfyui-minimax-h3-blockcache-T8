# MiniMax H3 Block Cache (T8)

[简体中文](README.md) | English

An experimental F1B0 Block Cache node for ComfyUI's native MiniMax H3 model. It always computes Block 0 for each model call and, when both target audio and target video are sufficiently stable, reuses the residual from the remaining blocks to skip Blocks 1–49 entirely.

## Installation

Search for `minimax-h3-block-cache-t8` or `MiniMax H3 Block Cache (T8)` in ComfyUI Manager, install it, and restart ComfyUI.

You can also clone the repository manually into `ComfyUI/custom_nodes`:

```powershell
git clone https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8.git
```

This node requires ComfyUI `0.30.0` or later and ComfyUI's native MiniMax H3 implementation. It has no additional Python dependencies.

## Usage

After restarting ComfyUI, add `MiniMax H3 Block Cache (T8)` from `advanced/model_patches`:

```text
Load Diffusion Model
        │
        ▼
MiniMax H3 Block Cache (T8)
        ├────────► Basic Scheduler
        └────────► Basic Guider
```

The default settings are a conservative starting point:

| Parameter | Default | Description |
| --- | ---: | --- |
| `residual_diff_threshold` | `0.12` | Higher values produce more cache hits but are also more likely to alter the result |
| `start_percent` | `0.08` | Before this point, all blocks run and the cache only warms up |
| `end_percent` | `0.95` | After this point, caching is disabled |
| `max_consecutive_hits` | `2` | Maximum consecutive cache hits before a mandatory refresh |
| `cache_device` | `cpu` | CPU saves VRAM; GPU reduces transfers but consumes VRAM |
| `metric_stride` | `8` | Sampling stride used by the audio/video stability metrics |

The node accepts only ComfyUI's native `MiniMaxH3Model`. It rejects EasyCache, LazyCache, Spectrum Apply MiniMax H3, the native Block Sparse Attention node, and existing H3 `double_block` replacements. Block Cache cannot be chained with Spectrum or native Block Sparse Attention: those nodes change H3 block execution or maintain per-block state without sharing Block Cache's skipping, replay, or invalidation contract. Regular SageAttention does not replace H3 blocks and can still be used as the attention backend. The node returns a cloned MODEL and does not modify the input model.

## Implementation Boundaries

- Audio and video Block 0 residual differences are evaluated independently. If either exceeds the threshold, all 50 blocks run.
- The cache stores only target audio/video segments, not text, conditioning, or reference rows.
- On a hit, the H3 block loop short-circuits after Block 0, then continues through the native H3 final layer and unpatchify path. Current ComfyUI passes the sigma, full sampling schedule, and audio/video shifts required by PDD; legacy ComfyUI keeps the four-argument final layer call.
- Current ComfyUI returns raw audio velocity; legacy ComfyUI retains the audio schedule slope.
- Full forwards preserve ComfyUI's dynamic vbar prefetch. Cache hits clean up unconsumed prefetch state and support both the current and legacy cleanup interfaces plus the current malloc-graph block scope.
- Cache-state allocations temporarily pause the active Comfy Compiler malloc graph so tensors retained across blocks or sampling steps do not enter the allocation graph. Legacy Core versions without that interface keep the original path.
- All tensor state is scoped to one sampling execution and is released after normal completion, errors, or cancellation.
- The node does not modify ComfyUI core files or Comfy Kitchen `.pyd` binaries and adds no dependencies or network requests.

This is an approximate cache and does not guarantee lossless output for the same seed. Prompt, resolution, frame count, step count, scheduler, attention backend, and quantization format can all affect cache hit rate and output quality.

## License

This project is licensed under the [Apache License 2.0](LICENSE). Personal and commercial use, modification, and redistribution are permitted subject to the license requirements, including preservation of the license text, change notices, and attribution notices.

## ComfyUI Version Compatibility

ComfyUI commit `bdcb886a4` moved MiniMax H3 to `ModelType.FLOW_AV` / `ModelSamplingAV` and changed H3 audio output from slope-scaled velocity to raw audio velocity. This node detects `ModelSamplingAV` at runtime:

- On current ComfyUI, a cache hit returns raw audio velocity and preserves the native outer H3 `audio_scale` / carry transform boundary.
- On legacy ComfyUI, the optional `time_shift_slope` helper preserves the previous slope-scaled velocity behavior.

The current path was validated on ComfyUI `a464ac335` with a custom-node whitelist cold start, a full startup, and a real FL2VA INT8 H3 forward at 512×512, 22 frames, and 1 step. Block Cache attached and executed successfully, reporting `cached 0/1 model forwards` and approximately 19.1 MiB of CPU cache. The legacy path has numerical coverage using a simulated old interface but has not yet been tested with a real model in a separate legacy ComfyUI checkout. A core `SaveLatent` / `NestedTensor.contiguous` error triggered by that test workflow after sampling completed was unrelated to Block Cache.

ComfyUI commit `efd4e951a` introduced CUDA Graph prefetch state and changed `cleanup_prefetched_modules` from one argument to `(module, comfy_modules)`. The node uses the simultaneously introduced `comfy.model_prefetch.GRAPH_MODULES` capability as the compatibility boundary: current ComfyUI receives both the prefetched block and its module list, while legacy ComfyUI continues to use the one-argument cleanup call.

ComfyUI commit `2504e68d4` added PDD LoRA sampling arguments to the H3 final layer. The node detects the final-layer protocol: current Core receives `sigma`, `sample_sigmas`, and the audio/video shifts on a cache hit, while legacy Core keeps the original four-argument call. Both regular H3 heads and PDD head banks continue through Core's own final-layer implementation.

ComfyUI commit `e308cc73b` combined Sol-Attn, top-k/SLA, and VSA/FastVideo in the native `Block Sparse Attention` node. It replaces every H3 DiT block and maintains cross-step per-block statistics, so this node rejects the combination before sampling in either connection order. After commit `804eb5513` introduced Comfy Compiler, cache allocations also pause its malloc graph and a cache hit closes the active block scope. Legacy Core versions without these interfaces do not call them.

Version `1.0.4` was checked against ComfyUI `fbed745c8` (`0.34.0`) with 22 tests, Ruff, compileall, a whitelist cold start, a current seven-argument final-layer hit probe, and both native Block Sparse Attention connection orders through the actual Core API. Additional import probes used independent checkouts at ComfyUI `a464ac335`, the previous release baseline `187eda8ef`, and pre-raw-audio commit `2eb609766`, confirming detection of the legacy four-argument final layer, both prefetch cleanup interfaces, the no-malloc-scope path, and the older slope-scaled audio protocol. To keep the 33B model test low-risk, this release did not repeat a full H3 generation; the latest real FL2VA forward remains the `a464ac335` baseline documented above.

## Local Smoke Benchmark

Environment: RTX 4060 Ti 16GB, MiniMax H3 FL2VA INT8 ConvRot, `res_multistep` with 20 steps, 256×160, 5 frames, and a fixed prompt and seed. Times cover only tqdm denoising and exclude text encoding, model loading, and VAE decoding.

| Configuration | Hits | Denoise | Relative to baseline |
| --- | ---: | ---: | ---: |
| No cache | 0/20 | 485.8s | 1.00× |
| CPU cache, default stride 8 | 7/20 | 443.8s | 1.09× |
| CPU cache, stride 9 probe | 8/20 | 403.2s | 1.20× |
| GPU cache, default stride 8 | 7/20 | 471.6s | 1.03× |

A cache-hit step took approximately 1–2 seconds, while a full step took approximately 24–39 seconds. This desktop environment showed substantial load variance, and the sample is much smaller than a production H3 video. These figures only demonstrate that a real net-acceleration path exists; they should not be generalized into a fixed advertised speedup. Stride 9 changes the boundary decision and is therefore not used as the default.

## Validation

```powershell
python -m unittest discover -s tests -v
python -m ruff check .
```

The current suite contains 22 tests covering joint audio/video decisions, actual block skipping, consecutive-hit limits, the sampling window, UUID/shape/sigma invalidation, independent target-only storage, execution cleanup, model cloning, patch conflicts, mutual exclusion for both connection orders of Spectrum and Block Sparse Attention, current and legacy prefetch/malloc-scope interfaces, Comfy Compiler cache-allocation isolation, current and legacy H3 final-layer protocols, H3 audio/video output shape and dtype, and both current raw and legacy slope-scaled audio velocity finalization.

Formal performance conclusions still require the audio/video quality matrix and repeated timing runs for the official 124-frame T2V, FL2VA, and Ref2VA workloads.
