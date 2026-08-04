from __future__ import annotations

from dataclasses import dataclass
import logging

import torch


CACHE_KEY = "minimax_h3_block_cache_t8"


@dataclass(frozen=True)
class H3BlockCacheConfig:
    residual_diff_threshold: float
    start_percent: float
    end_percent: float
    max_consecutive_hits: int
    cache_device: str
    metric_stride: int
    verbose: bool = False


@dataclass
class _CacheStream:
    metadata: tuple | None = None
    previous_audio_indicator: torch.Tensor | None = None
    previous_video_indicator: torch.Tensor | None = None
    audio_tail: torch.Tensor | None = None
    video_tail: torch.Tensor | None = None
    consecutive_hits: int = 0
    last_sigma: float | None = None

    def clear_tensors(self):
        self.previous_audio_indicator = None
        self.previous_video_indicator = None
        self.audio_tail = None
        self.video_tail = None
        self.consecutive_hits = 0


@dataclass
class _ForwardState:
    stream: _CacheStream
    audio_range: tuple[int, int]
    video_range: tuple[int, int]
    audio_before: torch.Tensor
    video_before: torch.Tensor
    sigma: float | None
    use_cache: bool = False
    update_cache: bool = True
    audio_anchor: torch.Tensor | None = None
    video_anchor: torch.Tensor | None = None


class H3BlockCacheHit(Exception):
    def __init__(self, hidden: torch.Tensor, t_emb: torch.Tensor, audio_segment: tuple, video_segment: tuple):
        super().__init__()
        self.hidden = hidden
        self.t_emb = t_emb
        self.audio_segment = audio_segment
        self.video_segment = video_segment


class H3BlockCache:
    def __init__(self, config: H3BlockCacheConfig, total_blocks: int):
        self.config = config
        self.total_blocks = total_blocks
        self.start_sigma = float("inf")
        self.end_sigma = 0.0
        self.streams: dict[tuple, _CacheStream] = {}
        self.current: _ForwardState | None = None
        self.total_forwards = 0
        self.full_forwards = 0
        self.cache_hits = 0

    def clone(self):
        return H3BlockCache(self.config, self.total_blocks)

    def prepare(self, model_sampling):
        self.start_sigma = float(model_sampling.percent_to_sigma(self.config.start_percent))
        self.end_sigma = float(model_sampling.percent_to_sigma(self.config.end_percent))
        return self

    def _sample(self, x: torch.Tensor, bounds: tuple[int, int]) -> torch.Tensor:
        start, end = bounds
        stride = self.config.metric_stride
        return self._copy_to_cache_device(x[start:end:stride, ::stride].float())

    @staticmethod
    def _stream_key(transformer_options: dict) -> tuple:
        uuids = transformer_options.get("uuids")
        if not uuids:
            return ("default",)
        return tuple(str(value) for value in uuids)

    @staticmethod
    def _sigma(transformer_options: dict) -> float | None:
        sigmas = transformer_options.get("sigmas")
        if sigmas is None:
            return None
        if isinstance(sigmas, torch.Tensor):
            return float(sigmas.flatten()[0])
        if isinstance(sigmas, (list, tuple)):
            return float(sigmas[0])
        return float(sigmas)

    @staticmethod
    def _target_ranges(x: torch.Tensor, mod_segments: list) -> tuple[tuple[int, int], tuple[int, int]]:
        if len(mod_segments) < 2:
            raise RuntimeError("MiniMax H3 Block Cache could not find target audio/video segments")
        audio_start, audio_end = int(mod_segments[-2][0]), int(mod_segments[-2][1])
        video_start, video_end = int(mod_segments[-1][0]), int(mod_segments[-1][1])
        if audio_start < 0 or audio_start >= audio_end or audio_end != video_start or video_end != x.shape[0]:
            raise RuntimeError("MiniMax H3 Block Cache received an unsupported packed sequence layout")
        return (audio_start, audio_end), (video_start, video_end)

    def _metadata(self, x: torch.Tensor, audio_range: tuple[int, int], video_range: tuple[int, int]) -> tuple:
        return (tuple(x.shape), x.dtype, x.device, audio_range, video_range)

    def _eligible(self, sigma: float | None) -> bool:
        if sigma is None:
            return False
        return self.end_sigma < sigma <= self.start_sigma

    def _past_end(self, sigma: float | None) -> bool:
        return sigma is not None and sigma <= self.end_sigma

    def begin_forward(self, x: torch.Tensor, mod_segments: list, transformer_options: dict):
        audio_range, video_range = self._target_ranges(x, mod_segments)
        key = self._stream_key(transformer_options)
        sigma = self._sigma(transformer_options)
        metadata = self._metadata(x, audio_range, video_range)
        stream = self.streams.setdefault(key, _CacheStream())

        if stream.metadata != metadata or (stream.last_sigma is not None and sigma is not None and sigma > stream.last_sigma):
            stream.clear_tensors()
            stream.metadata = metadata
        stream.last_sigma = sigma

        if self._past_end(sigma):
            stream.clear_tensors()

        self.current = _ForwardState(
            stream=stream,
            audio_range=audio_range,
            video_range=video_range,
            audio_before=self._sample(x, audio_range),
            video_before=self._sample(x, video_range),
            sigma=sigma,
            update_cache=not self._past_end(sigma),
        )
        self.total_forwards += 1

    @staticmethod
    def _relative_diff(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        denominator = previous.abs().mean().clamp_min(1e-6)
        return (current - previous).abs().mean() / denominator

    def finish_first_block(self, x: torch.Tensor):
        current = self.current
        if current is None:
            raise RuntimeError("MiniMax H3 Block Cache first block ran without forward state")

        audio_indicator = self._sample(x, current.audio_range) - current.audio_before
        video_indicator = self._sample(x, current.video_range) - current.video_before
        stream = current.stream

        can_compare = (
            stream.previous_audio_indicator is not None
            and stream.previous_video_indicator is not None
            and stream.audio_tail is not None
            and stream.video_tail is not None
        )
        audio_score = video_score = None
        if can_compare:
            scores = torch.stack((
                self._relative_diff(audio_indicator, stream.previous_audio_indicator),
                self._relative_diff(video_indicator, stream.previous_video_indicator),
            )).tolist()
            audio_score, video_score = float(scores[0]), float(scores[1])

        current.use_cache = (
            self._eligible(current.sigma)
            and can_compare
            and max(audio_score, video_score) < self.config.residual_diff_threshold
            and stream.consecutive_hits < self.config.max_consecutive_hits
        )

        if current.use_cache:
            stream.consecutive_hits += 1
            self.cache_hits += 1
        else:
            stream.previous_audio_indicator = audio_indicator
            stream.previous_video_indicator = video_indicator
            stream.audio_tail = None
            stream.video_tail = None
            stream.consecutive_hits = 0
            self.full_forwards += 1
            if current.update_cache:
                audio_start, audio_end = current.audio_range
                video_start, video_end = current.video_range
                current.audio_anchor = self._copy_to_cache_device(x[audio_start:audio_end])
                current.video_anchor = self._copy_to_cache_device(x[video_start:video_end])

        if self.config.verbose:
            if audio_score is None:
                logging.info("MiniMax H3 Block Cache - refresh (cache not ready), sigma=%s", current.sigma)
            else:
                logging.info(
                    "MiniMax H3 Block Cache - %s, audio_diff=%.5f, video_diff=%.5f, sigma=%s",
                    "hit" if current.use_cache else "refresh",
                    audio_score,
                    video_score,
                    current.sigma,
                )

    def _copy_to_cache_device(self, tensor: torch.Tensor) -> torch.Tensor:
        device = "cpu" if self.config.cache_device == "cpu" else tensor.device
        return tensor.detach().to(device=device, copy=True).contiguous()

    @staticmethod
    def _add_tail(x: torch.Tensor, bounds: tuple[int, int], tail: torch.Tensor):
        start, end = bounds
        x[start:end].add_(tail.to(device=x.device, dtype=x.dtype))

    def apply_cached_tail(self, x: torch.Tensor):
        current = self.current
        if current is None:
            raise RuntimeError("MiniMax H3 Block Cache hit without forward state")
        self._add_tail(x, current.audio_range, current.stream.audio_tail)
        self._add_tail(x, current.video_range, current.stream.video_tail)
        self.current = None

    def finish_last_block(self, output: dict):
        current = self.current
        if current is None:
            raise RuntimeError("MiniMax H3 Block Cache last block ran without forward state")

        if current.update_cache:
            x = output["img"]
            audio_start, audio_end = current.audio_range
            video_start, video_end = current.video_range
            audio_tail = self._copy_to_cache_device(x[audio_start:audio_end])
            video_tail = self._copy_to_cache_device(x[video_start:video_end])
            audio_tail.sub_(current.audio_anchor)
            video_tail.sub_(current.video_anchor)
            current.stream.audio_tail = audio_tail
            current.stream.video_tail = video_tail
        self.current = None

    def cache_bytes(self) -> int:
        total = 0
        for stream in self.streams.values():
            for tensor in (stream.audio_tail, stream.video_tail):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return total

    def reset(self):
        self.current = None
        self.streams.clear()


class H3BlockPatch:
    def __init__(self, block_index: int):
        self.block_index = block_index

    def __call__(self, args: dict, extra_options: dict):
        cache = args["transformer_options"].get(CACHE_KEY)
        if not isinstance(cache, H3BlockCache):
            raise RuntimeError("MiniMax H3 Block Cache runtime state is missing")

        if self.block_index == 0:
            cache.begin_forward(args["img"], args["mod_segments"], args["transformer_options"])
            output = extra_options["original_block"](args)
            cache.finish_first_block(output["img"])
            if cache.current.use_cache:
                cache.apply_cached_tail(output["img"])
                audio_start, audio_end, audio_tag = args["mod_segments"][-2]
                video_start, video_end, video_tag = args["mod_segments"][-1]
                raise H3BlockCacheHit(
                    output["img"],
                    args["t_emb"],
                    (audio_start, audio_end, audio_tag // 3),
                    (video_start, video_end, video_tag // 3),
                )
            return output

        output = extra_options["original_block"](args)
        if self.block_index == cache.total_blocks - 1:
            cache.finish_last_block(output)
        return output
