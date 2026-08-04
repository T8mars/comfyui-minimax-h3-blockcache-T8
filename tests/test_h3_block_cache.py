import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from h3_block_cache import H3BlockCache, H3BlockCacheConfig, H3BlockCacheHit, H3BlockPatch


class _Sampling:
    @staticmethod
    def percent_to_sigma(percent):
        return 1.0 - percent


def _config(**overrides):
    values = {
        "residual_diff_threshold": 0.12,
        "start_percent": 0.0,
        "end_percent": 1.0,
        "max_consecutive_hits": 2,
        "cache_device": "cpu",
        "metric_stride": 1,
        "verbose": False,
    }
    values.update(overrides)
    return H3BlockCacheConfig(**values)


class BlockChain:
    def __init__(self, cache, multipliers):
        self.cache = cache
        self.patches = {
            0: H3BlockPatch(0),
            len(multipliers) - 1: H3BlockPatch(len(multipliers) - 1),
        }
        self.multipliers = multipliers
        self.calls = [0] * len(multipliers)
        self.last_hit = None

    def run(self, x, sigma=0.5, uuid="cond"):
        transformer_options = {
            "minimax_h3_block_cache_t8": self.cache,
            "sigmas": torch.tensor([sigma]),
            "uuids": [uuid],
        }
        mod_segments = [(0, 1, 1), (1, 3, 5), (3, x.shape[0], 0)]
        for index in range(len(self.multipliers)):
            args = {
                "img": x,
                "t_emb": None,
                "mod_segments": mod_segments,
                "rope_freqs": None,
                "transformer_options": transformer_options,
            }

            def original(current_args, block_index=index):
                self.calls[block_index] += 1
                current_args["img"].add_(current_args["img"] * self.multipliers[block_index] + block_index + 1)
                return {"img": current_args["img"]}

            patch = self.patches.get(index)
            if patch is None:
                x = original(args)["img"]
                continue
            try:
                x = patch(args, {"original_block": original})["img"]
            except H3BlockCacheHit as hit:
                self.last_hit = hit
                x = hit.hidden
                break
        return x


class H3BlockCacheTests(unittest.TestCase):
    def test_cache_hit_skips_all_blocks_after_first(self):
        cache = H3BlockCache(_config(), total_blocks=4).prepare(_Sampling())
        chain = BlockChain(cache, [0.1, 0.0, 0.0, 0.0])
        baseline = chain.run(torch.ones(6, 8))
        cached = chain.run(torch.ones(6, 8))

        self.assertTrue(torch.equal(cached[1:], baseline[1:]))
        self.assertEqual(chain.calls, [2, 1, 1, 1])
        self.assertEqual(cache.cache_hits, 1)
        self.assertEqual(cache.full_forwards, 1)
        self.assertEqual(chain.last_hit.audio_segment, (1, 3, 1))
        self.assertEqual(chain.last_hit.video_segment, (3, 6, 0))

    def test_audio_change_forces_refresh_when_video_is_stable(self):
        cache = H3BlockCache(_config(), total_blocks=3).prepare(_Sampling())
        chain = BlockChain(cache, [0.25, 0.0, 0.0])
        chain.run(torch.ones(6, 4))
        changed = torch.ones(6, 4)
        changed[1:3] *= 4
        chain.run(changed)

        self.assertEqual(cache.cache_hits, 0)
        self.assertEqual(chain.calls, [2, 2, 2])

    def test_max_consecutive_hits_forces_refresh(self):
        cache = H3BlockCache(_config(max_consecutive_hits=1), total_blocks=3).prepare(_Sampling())
        chain = BlockChain(cache, [0.1, 0.0, 0.0])
        chain.run(torch.ones(6, 4))
        chain.run(torch.ones(6, 4))
        chain.run(torch.ones(6, 4))

        self.assertEqual(cache.cache_hits, 1)
        self.assertEqual(cache.full_forwards, 2)
        self.assertEqual(chain.calls, [3, 2, 2])

    def test_cache_stores_target_rows_only_on_cpu(self):
        cache = H3BlockCache(_config(), total_blocks=3).prepare(_Sampling())
        chain = BlockChain(cache, [0.1, 0.0, 0.0])
        chain.run(torch.ones(8, 4))
        stream = next(iter(cache.streams.values()))

        self.assertEqual(tuple(stream.audio_tail.shape), (2, 4))
        self.assertEqual(tuple(stream.video_tail.shape), (5, 4))
        self.assertEqual(stream.audio_tail.device.type, "cpu")
        self.assertEqual(stream.video_tail.device.type, "cpu")
        self.assertEqual(stream.audio_tail.untyped_storage().nbytes(), stream.audio_tail.numel() * stream.audio_tail.element_size())
        self.assertEqual(stream.video_tail.untyped_storage().nbytes(), stream.video_tail.numel() * stream.video_tail.element_size())
        self.assertEqual(stream.previous_audio_indicator.device.type, "cpu")
        self.assertEqual(stream.previous_video_indicator.device.type, "cpu")

    def test_sampling_window_forces_full_forwards_outside_range(self):
        cache = H3BlockCache(_config(start_percent=0.2, end_percent=0.8), total_blocks=3).prepare(_Sampling())
        chain = BlockChain(cache, [0.1, 0.0, 0.0])
        chain.run(torch.ones(6, 4), sigma=0.9)
        chain.run(torch.ones(6, 4), sigma=0.85)

        self.assertEqual(cache.cache_hits, 0)
        self.assertEqual(chain.calls, [2, 2, 2])

        chain.run(torch.ones(6, 4), sigma=0.1)
        stream = next(iter(cache.streams.values()))
        self.assertIsNone(stream.audio_tail)
        self.assertIsNone(stream.video_tail)

    def test_uuid_streams_are_isolated(self):
        cache = H3BlockCache(_config(), total_blocks=3).prepare(_Sampling())
        chain = BlockChain(cache, [0.1, 0.0, 0.0])
        chain.run(torch.ones(6, 4), uuid="positive")
        chain.run(torch.ones(6, 4), uuid="negative")
        chain.run(torch.ones(6, 4), uuid="positive")

        self.assertEqual(cache.cache_hits, 1)
        self.assertEqual(len(cache.streams), 2)
        self.assertEqual(chain.calls, [3, 2, 2])

    def test_sigma_reversal_invalidates_stream(self):
        cache = H3BlockCache(_config(), total_blocks=3).prepare(_Sampling())
        chain = BlockChain(cache, [0.1, 0.0, 0.0])
        chain.run(torch.ones(6, 4), sigma=0.5)
        chain.run(torch.ones(6, 4), sigma=0.6)

        self.assertEqual(cache.cache_hits, 0)
        self.assertEqual(chain.calls, [2, 2, 2])

    def test_shape_change_invalidates_cache(self):
        cache = H3BlockCache(_config(), total_blocks=3).prepare(_Sampling())
        chain = BlockChain(cache, [0.1, 0.0, 0.0])
        chain.run(torch.ones(6, 4))
        chain.run(torch.ones(7, 4))

        self.assertEqual(cache.cache_hits, 0)
        self.assertEqual(chain.calls, [2, 2, 2])

    def test_reset_releases_execution_state(self):
        cache = H3BlockCache(_config(), total_blocks=3).prepare(_Sampling())
        chain = BlockChain(cache, [0.1, 0.0, 0.0])
        chain.run(torch.ones(6, 4))
        self.assertGreater(cache.cache_bytes(), 0)

        cache.reset()

        self.assertEqual(cache.streams, {})
        self.assertIsNone(cache.current)


if __name__ == "__main__":
    unittest.main()
