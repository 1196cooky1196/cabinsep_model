"""Model regressions: press Run, or discover with Python's unittest runner."""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from config import CabinSepConfig
from model import (
    CabinSep,
    CausalConformerBlock,
    CausalGlobalLayerNormMap,
    CausalGlobalLayerNormSequence,
    FullSubModule,
    StreamingMVDR,
    TACBlock,
)


def tiny_config(**overrides: object) -> CabinSepConfig:
    settings = dict(
        encoder_hidden_channels=4, encoder_output_channels=4,
        fusion_channels=8, fullband_conv_channels=2, fullband_lstm_hidden=16,
        subband_hidden=4, conformer_heads=2, conformer_ffn_dim=4,
        conformer_layers=1, mask_noise_hidden=8,
    )
    settings.update(overrides)
    return CabinSepConfig.small(**settings)


class AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class ArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)

    def test_reproduction_defaults_match_table_and_referenced_conformer(self) -> None:
        config = CabinSepConfig.small()
        self.assertEqual(config.tac_axis, "frequency")
        self.assertFalse(config.time_skip)
        self.assertTrue(config.conformer_relative_position)
        self.assertEqual(config.conformer_conv_kernel, 32)
        self.assertEqual(config.conformer_conv_norm, "batch")
        self.assertEqual(config.conformer_dropout, 0.1)
        model = CabinSep(config)
        module = model.full_sub_modules[0]
        self.assertEqual(module.tac.axis, "frequency")
        self.assertIsInstance(module.sub_band.conformer[0].convolution.batch_norm, nn.BatchNorm1d)

    def test_tac_compresses_channels_independently_at_each_frequency(self) -> None:
        tac = TACBlock(8, 13, 4, axis="channel")
        x = torch.randn(2, 8, 5, 13, requires_grad=True)
        permutation = torch.randperm(13)
        actual = tac(x[..., permutation])
        expected = tac(x)[..., permutation]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tac.linear_a.in_features, 8)
        self.assertEqual(tac.linear_a.out_features, 2)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        # Paper C/d weights must not depend on the FFT's number of bins.
        wider = TACBlock(8, 29, 4, axis="channel")
        wider.load_state_dict(tac.state_dict(), strict=True)
        self.assertEqual(wider(torch.randn(1, 8, 3, 29)).shape, (1, 8, 3, 29))

    def test_legacy_frequency_tac_keeps_old_shapes(self) -> None:
        tac = TACBlock(8, 13, 4, axis="frequency")
        self.assertEqual(tac.linear_a.weight.shape, (3, 13))
        self.assertEqual(tac.linear_c.weight.shape, (13, 6))
        self.assertEqual(tac(torch.randn(1, 8, 3, 13)).shape, (1, 8, 3, 13))

    def test_time_skip_preserves_parity_for_single_frame_prefix(self) -> None:
        module = FullSubModule(tiny_config(time_skip=True, inference_time_skip_offset=1)).eval()
        module.tac = AddOne()
        x = torch.zeros(1, 8, 5, 257)
        transformed = module._time_skip_tac(x)
        self.assertEqual(transformed[:, :, 0::2].sum().item(), 0.0)
        self.assertTrue((transformed[:, :, 1::2] == 1.0).all())
        torch.testing.assert_close(module._time_skip_tac(x[:, :, :1]), transformed[:, :, :1])
        module.time_skip = False
        self.assertTrue((module._time_skip_tac(x) == 1.0).all())

    def test_relative_attention_matches_independent_scalar_formula(self) -> None:
        block = CausalConformerBlock(4, 3, 2, 3, 0.0, 3, True).double().eval()
        with torch.no_grad():
            block.content_bias.normal_()
            block.position_bias.normal_()
        x = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
        mask = block._attention_mask(5, x.device)
        actual = block._relative_attention(x, mask)
        # Direct Eq. (8), Transformer-XL, with explicit (query-key) lag.
        projected = x @ block.attention.in_proj_weight.T + block.attention.in_proj_bias
        query, key, value = [part.reshape(5, 2, 2) for part in projected.chunk(3, dim=-1)]
        expected_frames = []
        for frame in range(5):
            expected_heads = []
            for head in range(2):
                logits, values = [], []
                for previous in range(max(0, frame - 2), frame + 1):
                    lag = frame - previous
                    position = torch.tensor(
                        [math.sin(lag), math.cos(lag), math.sin(lag / 100.0), math.cos(lag / 100.0)],
                        dtype=torch.float64,
                    )
                    relative = (position @ block.relative_projection.weight.T).reshape(2, 2)
                    content_score = ((query[frame, head] + block.content_bias[head]) * key[previous, head]).sum()
                    position_score = ((query[frame, head] + block.position_bias[head]) * relative[head]).sum()
                    logits.append((content_score + position_score) / math.sqrt(2.0))
                    values.append(value[previous, head])
                weights = torch.stack(logits).softmax(dim=0)
                expected_heads.append((weights[:, None] * torch.stack(values)).sum(dim=0))
            expected_frames.append(torch.cat(expected_heads))
        expected = block.attention.out_proj(torch.stack(expected_frames)[None])
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(block.relative_projection.weight.grad).all())

    def test_relative_attention_reduces_to_mha_without_position_terms(self) -> None:
        block = CausalConformerBlock(8, 4, 2, 3, 0.0, None, True).eval()
        with torch.no_grad():
            block.relative_projection.weight.zero_()
        x = torch.randn(3, 7, 8)
        mask = block._attention_mask(x.shape[1], x.device)
        actual = block._relative_attention(x, mask)
        expected, _ = block.attention(x, x, x, attn_mask=mask, need_weights=False)
        torch.testing.assert_close(actual, expected)

    def test_full_mask_path_is_causal_with_optional_skip_and_context(self) -> None:
        mixture = torch.randn(1, 4, 1536) * 0.1
        for skip in (False, True):
            model = CabinSep(tiny_config(
                time_skip=skip, inference_time_skip_offset=1,
                conformer_left_context_frames=3,
            )).eval()
            with torch.no_grad():
                full = model.estimate_masks(mixture)
                for length in (512, 1024):
                    prefix = model.estimate_masks(mixture[..., :length])
                    frames = prefix["speech_mask"].shape[2]
                    for name in ("speech_mask", "noise_mask"):
                        torch.testing.assert_close(
                            prefix[name], full[name][:, :, :frames], atol=1e-5, rtol=1e-5,
                        )

    def test_half_precision_cumulative_norm_does_not_overflow(self) -> None:
        sequence = torch.randn(1, 200, 512).half()
        norm = CausalGlobalLayerNormSequence(512)
        actual = norm(sequence)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), norm(sequence.float()), atol=2e-3, rtol=2e-3)
        feature_map = torch.randn(1, 16, 200, 52).half()
        map_norm = CausalGlobalLayerNormMap(16)
        actual = map_norm(feature_map)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), map_norm(feature_map.float()), atol=2e-3, rtol=2e-3)

    def test_stft_roundtrip_short_and_unaligned_audio(self) -> None:
        model = CabinSep(tiny_config())
        for length in (1, 511, 512, 513, 1001, 2048):
            waveform = torch.randn(1, 4, length) * 0.1
            reconstructed = model.istft(model.stft(waveform), length)
            torch.testing.assert_close(reconstructed, waveform, atol=2e-6, rtol=2e-5)
        half = torch.randn(1, 4, 512).half()
        self.assertEqual(model.stft(half).dtype, torch.complex64)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            model.stft(torch.zeros(1, 4, 0))


class MVDRTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(72)

    def test_covariance_state_chunking_matches_full_sequence(self) -> None:
        spectrum = torch.randn(2, 4, 9, 5, dtype=torch.complex64)
        speech = torch.rand(2, 3, 9, 5)
        noise = torch.rand_like(speech)
        mvdr = StreamingMVDR(4, forgetting_factor=0.93)
        references = [3, 1, 0]
        full, full_state = mvdr(spectrum, speech, noise, reference_microphones=references)
        first, state = mvdr(spectrum[:, :, :4], speech[:, :, :4], noise[:, :, :4], reference_microphones=references)
        second, state = mvdr(spectrum[:, :, 4:], speech[:, :, 4:], noise[:, :, 4:], state, references)
        torch.testing.assert_close(torch.cat((first, second), dim=2), full)
        torch.testing.assert_close(state.speech_numerator, full_state.speech_numerator)
        torch.testing.assert_close(state.noise_weight, full_state.noise_weight)

    def test_rank_one_signal_preserves_selected_reference_phase(self) -> None:
        source = torch.randn(1, 1, 5, 3, dtype=torch.complex128)
        steering = torch.tensor([1.0, 2.0j, -0.5 + 0.3j, 1.0 - 1.0j], dtype=torch.complex128)
        spectrum = steering[None, :, None, None] * source
        masks = torch.ones(1, 2, 5, 3, dtype=torch.float64)
        actual, _ = StreamingMVDR(4)(spectrum, masks, masks, reference_microphones=[2, 1])
        torch.testing.assert_close(actual, spectrum[:, [2, 1]], atol=1e-8, rtol=1e-8)

    def test_energetic_identical_channels_with_zero_requested_loading(self) -> None:
        spectrum = torch.ones(1, 4, 3, 2, dtype=torch.complex64) * 1000.0
        masks = torch.ones(1, 4, 3, 2)
        actual, _ = StreamingMVDR(4, diagonal_loading=0.0)(spectrum, masks, masks)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, spectrum, atol=0.2, rtol=2e-4)

    def test_zero_speech_mask_falls_back_to_reference_and_silence_is_finite(self) -> None:
        spectrum = torch.randn(1, 4, 3, 2, dtype=torch.complex64)
        masks = torch.zeros(1, 4, 3, 2)
        mvdr = StreamingMVDR(4)
        actual, _ = mvdr(spectrum, masks, masks)
        torch.testing.assert_close(actual, spectrum)
        actual, _ = mvdr(torch.zeros_like(spectrum), masks, masks)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual.abs().sum().item(), 0.0)

    def test_invalid_mvdr_inputs_fail_with_useful_errors(self) -> None:
        spectrum = torch.randn(1, 4, 3, 2, dtype=torch.complex64)
        masks = torch.ones(1, 4, 3, 2)
        mvdr = StreamingMVDR(4)
        _, state = mvdr(spectrum, masks, masks)
        with self.assertRaisesRegex(ValueError, "dtype/device"):
            mvdr(spectrum.to(torch.complex128), masks.double(), masks.double(), state)
        masks[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            mvdr(spectrum, masks, masks)
        with self.assertRaisesRegex(ValueError, "forgetting_factor"):
            StreamingMVDR(4, forgetting_factor=1.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
