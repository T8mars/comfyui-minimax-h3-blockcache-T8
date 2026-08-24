import importlib
import pathlib
import sys
import types
import unittest

import torch


COMFY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

node_module = importlib.import_module("custom_nodes.comfyui-minimax-h3-blockcache-T8.nodes")
cache_module = importlib.import_module("custom_nodes.comfyui-minimax-h3-blockcache-T8.h3_block_cache")

import comfy.model_patcher
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model


class _FakeModelPatcher:
    def __init__(self, diffusion_model):
        self.model = types.SimpleNamespace(diffusion_model=diffusion_model)
        self.model_options = {"transformer_options": {}}

    def clone(self):
        cloned = _FakeModelPatcher(self.model.diffusion_model)
        cloned.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        return cloned

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = comfy.model_patcher.set_model_options_patch_replace(
            self.model_options, patch, name, block_name, number, transformer_index
        )

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        comfy.patcher_extension.add_wrapper_with_key(wrapper_type, key, wrapper, self.model_options, is_model_options=True)


def _tiny_h3(blocks=4):
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(blocks))
    return model


class NodeTests(unittest.TestCase):
    def _assert_prefetch_cleanup(self, takes_module):
        block = object()
        comfy_modules = [object()]
        queue = [(None, (block, comfy_modules))]
        model = types.SimpleNamespace(blocks=[block])
        calls = []
        original_queues = comfy.model_prefetch.PREFETCH_QUEUES
        original_cleanup = comfy.model_prefetch.cleanup_prefetched_modules
        original_takes_module = node_module.PREFETCH_CLEANUP_TAKES_MODULE

        if takes_module:
            def cleanup(module, modules):
                calls.append((module, modules))
        else:
            def cleanup(modules):
                calls.append(modules)

        try:
            comfy.model_prefetch.PREFETCH_QUEUES = [queue]
            comfy.model_prefetch.cleanup_prefetched_modules = cleanup
            node_module.PREFETCH_CLEANUP_TAKES_MODULE = takes_module
            node_module._cleanup_short_circuited_prefetch(model)
        finally:
            comfy.model_prefetch.PREFETCH_QUEUES = original_queues
            comfy.model_prefetch.cleanup_prefetched_modules = original_cleanup
            node_module.PREFETCH_CLEANUP_TAKES_MODULE = original_takes_module

        self.assertEqual(queue, [None])
        self.assertEqual(len(calls), 1)
        if takes_module:
            self.assertIs(calls[0][0], block)
            self.assertIs(calls[0][1], comfy_modules)
        else:
            self.assertIs(calls[0], comfy_modules)

    def test_node_clones_model_and_patches_only_boundary_blocks(self):
        original = _FakeModelPatcher(_tiny_h3())

        patched = node_module.MiniMaxH3BlockCacheNode.execute(
            original, 0.12, 0.08, 0.95, 2, "cpu", 8, False
        )[0]

        self.assertEqual(original.model_options, {"transformer_options": {}})
        transformer_options = patched.model_options["transformer_options"]
        replacements = transformer_options["patches_replace"]["dit"]
        self.assertEqual(set(replacements), {("double_block", 0), ("double_block", 3)})
        wrappers = transformer_options["wrappers"]
        self.assertIn(node_module.WRAPPER_KEY, wrappers[comfy.patcher_extension.WrappersMP.OUTER_SAMPLE])
        self.assertIn(node_module.WRAPPER_KEY, wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL])
        self.assertNotIn("prefetch_dynamic_vbars", transformer_options)

    def test_node_rejects_existing_block_patch(self):
        model = _FakeModelPatcher(_tiny_h3())
        model.set_model_patch_replace(object(), "dit", "double_block", 2)

        with self.assertRaisesRegex(ValueError, "already patched"):
            node_module.MiniMaxH3BlockCacheNode.execute(model, 0.12, 0.08, 0.95, 2, "cpu", 8, False)

    def test_node_rejects_spectrum_applied_before_block_cache(self):
        model = _FakeModelPatcher(_tiny_h3())
        model.model_options[node_module.SPECTRUM_BINDING_KEY] = object()

        with self.assertRaisesRegex(ValueError, "Spectrum Apply MiniMax H3"):
            node_module.MiniMaxH3BlockCacheNode.execute(model, 0.12, 0.08, 0.95, 2, "cpu", 8, False)

    def test_sample_wrapper_rejects_spectrum_applied_after_block_cache(self):
        original_model_options = {
            node_module.SPECTRUM_BINDING_KEY: object(),
            "transformer_options": {},
        }
        guider = types.SimpleNamespace(model_options=original_model_options)

        class _Executor:
            class_obj = guider

            def __call__(self, *args, **kwargs):
                raise AssertionError("sampling must not start when Spectrum is active")

        with self.assertRaisesRegex(RuntimeError, "Spectrum Apply MiniMax H3"):
            node_module.h3_block_cache_sample_wrapper(_Executor())
        self.assertIs(guider.model_options, original_model_options)

    def test_current_prefetch_cleanup_passes_prefetched_module(self):
        self._assert_prefetch_cleanup(takes_module=True)

    def test_legacy_prefetch_cleanup_keeps_single_argument(self):
        self._assert_prefetch_cleanup(takes_module=False)

    def test_short_circuit_finalizer_preserves_h3_output_shapes(self):
        class _FinalLayer:
            def __call__(self, hidden, t_emb, video_segment, audio_segment):
                return torch.arange(32, dtype=torch.float32).reshape(8, 4), torch.arange(12, dtype=torch.float32).reshape(4, 3)

        model = types.SimpleNamespace(
            blocks=[],
            final_layer=_FinalLayer(),
            patch_size=(1, 2, 2),
            latents_dim=1,
            sigma_shift_video=12.0,
            sigma_shift_audio=3.0,
        )
        hit = cache_module.H3BlockCacheHit(torch.empty(12, 4), torch.empty(1, 4), (0, 4, 0), (4, 12, 0))

        class _Executor:
            class_obj = model

            def __call__(self, *args, **kwargs):
                raise hit

        video = torch.empty(1, 1, 2, 4, 4)
        audio = torch.empty(1, 3, 2, 2)
        output = node_module.h3_block_cache_diffusion_wrapper(
            _Executor(), [video, audio], torch.tensor([1000.0]), None, {}
        )

        self.assertEqual(tuple(output[0].shape), tuple(video.shape))
        self.assertEqual(tuple(output[1].shape), tuple(audio.shape))
        self.assertEqual(output[0].dtype, video.dtype)
        self.assertEqual(output[1].dtype, audio.dtype)

    def test_current_h3_short_circuit_returns_raw_audio_velocity(self):
        audio_out = torch.tensor([1.0, -2.0])
        result = node_module._finalize_audio_velocity(
            audio_out,
            torch.tensor(0.5),
            12.0,
            3.0,
            raw_audio_velocity=True,
        )
        self.assertTrue(torch.equal(result, -audio_out))

    def test_legacy_h3_short_circuit_keeps_slope_scaled_velocity(self):
        original = node_module._legacy_time_shift_slope
        node_module._legacy_time_shift_slope = lambda *_args: torch.tensor(0.25)
        try:
            audio_out = torch.tensor([4.0, -8.0])
            result = node_module._finalize_audio_velocity(
                audio_out,
                torch.tensor(0.5),
                12.0,
                3.0,
                raw_audio_velocity=False,
            )
        finally:
            node_module._legacy_time_shift_slope = original
        self.assertTrue(torch.equal(result, torch.tensor([-1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
