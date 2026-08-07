from __future__ import annotations

import logging

from comfy_api.latest import io, ComfyExtension
import comfy.ldm.common_dit
import comfy.model_sampling
import comfy.model_patcher
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model, unpack_audio, unpatchify_video

try:
    # Present in legacy ComfyUI H3, removed when H3 switched to raw audio velocity.
    from comfy.ldm.minimax.model import time_shift_slope as _legacy_time_shift_slope
except ImportError:  # Current ComfyUI
    _legacy_time_shift_slope = None

from .h3_block_cache import CACHE_KEY, H3BlockCache, H3BlockCacheConfig, H3BlockCacheHit, H3BlockPatch


WRAPPER_KEY = "minimax_h3_block_cache_t8"
H3_RETURNS_RAW_AUDIO_VELOCITY = hasattr(comfy.model_sampling, "ModelSamplingAV")


def _finalize_audio_velocity(
    audio_out,
    sigma_video,
    shift_video: float,
    shift_audio: float,
    *,
    raw_audio_velocity: bool,
):
    if raw_audio_velocity:
        return -audio_out
    if _legacy_time_shift_slope is None:
        raise RuntimeError("Legacy MiniMax H3 audio slope helper is unavailable")
    audio_slope = _legacy_time_shift_slope(
        sigma_video, shift_video, shift_audio
    ).to(audio_out.dtype)
    return (-audio_slope) * audio_out


def h3_block_cache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original_model_options = guider.model_options
    runtime_cache = None
    try:
        transformer_options = original_model_options["transformer_options"]
        if "easycache" in transformer_options:
            raise RuntimeError("MiniMax H3 Block Cache cannot be combined with EasyCache or LazyCache")

        prototype = transformer_options[CACHE_KEY]
        guider.model_options = comfy.model_patcher.create_model_options_clone(original_model_options)
        runtime_cache = prototype.clone().prepare(guider.model_patcher.model.model_sampling)
        guider.model_options["transformer_options"][CACHE_KEY] = runtime_cache
        return executor(*args, **kwargs)
    finally:
        if runtime_cache is not None:
            logging.info(
                "MiniMax H3 Block Cache - cached %d/%d model forwards, cache %.1f MiB on %s",
                runtime_cache.cache_hits,
                runtime_cache.total_forwards,
                runtime_cache.cache_bytes() / (1024 * 1024),
                runtime_cache.config.cache_device,
            )
            runtime_cache.reset()
        guider.model_options = original_model_options


def _cleanup_short_circuited_prefetch(model: MiniMaxH3Model):
    block_ids = {id(block) for block in model.blocks}
    for queue in reversed(comfy.model_prefetch.PREFETCH_QUEUES):
        belongs_to_model = False
        for entry in queue:
            module = entry[1][0] if isinstance(entry, tuple) else entry
            if id(module) in block_ids:
                belongs_to_model = True
                break
        if not belongs_to_model:
            continue
        for entry in queue:
            if not isinstance(entry, tuple):
                continue
            comfy_modules = entry[1][1]
            if comfy_modules is not None:
                comfy.model_prefetch.cleanup_prefetched_modules(comfy_modules)
        queue[:] = [None]
        return


def h3_block_cache_diffusion_wrapper(executor, *args, **kwargs):
    try:
        return executor(*args, **kwargs)
    except H3BlockCacheHit as hit:
        model = executor.class_obj
        _cleanup_short_circuited_prefetch(model)

        video_x, audio_x = args[0][0], args[0][1]
        timestep = args[1]
        transformer_options = args[3]
        orig_t, orig_h, orig_w = video_x.shape[2:]
        padded_video = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
        latent_t, latent_h, latent_w = padded_video.shape[2:]

        video_rows, audio_rows = model.final_layer(hit.hidden, hit.t_emb, hit.video_segment, hit.audio_segment)
        video_out = unpatchify_video(
            video_rows,
            latent_t,
            latent_h // model.patch_size[1],
            latent_w // model.patch_size[2],
            model.latents_dim,
            model.patch_size,
        )
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(audio_rows)

        sigma_video = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        shift_video = float(transformer_options.get("minimax_h3_sigma_shift_video", model.sigma_shift_video))
        shift_audio = float(transformer_options.get("minimax_h3_sigma_shift_audio", model.sigma_shift_audio))
        audio_velocity = _finalize_audio_velocity(
            audio_out,
            sigma_video,
            shift_video,
            shift_audio,
            raw_audio_velocity=H3_RETURNS_RAW_AUDIO_VELOCITY,
        )
        return [-video_out.to(video_x.dtype), audio_velocity.to(audio_x.dtype)]


class MiniMaxH3BlockCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3BlockCacheT8",
            display_name="MiniMax H3 Block Cache (T8)",
            description="H3-specific F1B0 residual cache. Recomputes the first block and skips later blocks when both audio and video are stable.",
            category="advanced/model_patches",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="A native MiniMax H3 diffusion model."),
                io.Float.Input("residual_diff_threshold", min=0.0, default=0.12, max=1.0, step=0.01, tooltip="Higher values cache more model forwards and may change the result."),
                io.Float.Input("start_percent", min=0.0, default=0.08, max=1.0, step=0.01, tooltip="Sampling progress before which the cache stays in warmup.", advanced=True),
                io.Float.Input("end_percent", min=0.0, default=0.95, max=1.0, step=0.01, tooltip="Sampling progress after which the cache is disabled.", advanced=True),
                io.Int.Input("max_consecutive_hits", min=1, default=2, max=10, step=1, tooltip="Maximum cached forwards before a mandatory refresh.", advanced=True),
                io.Combo.Input("cache_device", options=["cpu", "gpu"], default="cpu", tooltip="CPU saves VRAM; GPU avoids residual transfers.", advanced=True),
                io.Int.Input("metric_stride", min=1, default=8, max=32, step=1, tooltip="Stride used for the audio/video stability metric. Lower values use more memory.", advanced=True),
                io.Boolean.Input("verbose", default=False, tooltip="Log per-forward audio/video cache scores.", advanced=True),
            ],
            outputs=[io.Model.Output(tooltip="The cloned MiniMax H3 model with Block Cache.")],
        )

    @classmethod
    def execute(
        cls,
        model: io.Model.Type,
        residual_diff_threshold: float,
        start_percent: float,
        end_percent: float,
        max_consecutive_hits: int,
        cache_device: str,
        metric_stride: int,
        verbose: bool,
    ) -> io.NodeOutput:
        if start_percent >= end_percent:
            raise ValueError("MiniMax H3 Block Cache start_percent must be lower than end_percent")

        diffusion_model = model.model.diffusion_model
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("MiniMax H3 Block Cache requires a native MiniMax H3 diffusion model")

        total_blocks = len(diffusion_model.blocks)
        if total_blocks < 2:
            raise ValueError("MiniMax H3 Block Cache requires at least two DiT blocks")

        transformer_options = model.model_options["transformer_options"]
        if "easycache" in transformer_options:
            raise ValueError("MiniMax H3 Block Cache cannot be combined with EasyCache or LazyCache")
        existing_replacements = transformer_options.get("patches_replace", {}).get("dit", {})
        conflicts = [("double_block", index) for index in range(total_blocks) if ("double_block", index) in existing_replacements]
        if conflicts:
            raise ValueError("MiniMax H3 Block Cache cannot replace DiT blocks that are already patched")

        model = model.clone()
        config = H3BlockCacheConfig(
            residual_diff_threshold=residual_diff_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            max_consecutive_hits=max_consecutive_hits,
            cache_device=cache_device,
            metric_stride=metric_stride,
            verbose=verbose,
        )
        transformer_options = model.model_options["transformer_options"].copy()
        transformer_options[CACHE_KEY] = H3BlockCache(config, total_blocks)
        model.model_options["transformer_options"] = transformer_options

        model.set_model_patch_replace(H3BlockPatch(0), "dit", "double_block", 0)
        model.set_model_patch_replace(H3BlockPatch(total_blocks - 1), "dit", "double_block", total_blocks - 1)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, h3_block_cache_sample_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, h3_block_cache_diffusion_wrapper)
        return io.NodeOutput(model)


class MiniMaxH3BlockCacheExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MiniMaxH3BlockCacheNode]


def comfy_entrypoint():
    return MiniMaxH3BlockCacheExtension()
