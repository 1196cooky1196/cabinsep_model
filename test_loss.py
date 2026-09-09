"""Regression checks for CabinSep's padded waveform losses.

Open this file and press Run, or run unittest discovery. Tests use short CPU
signals and require neither a dataset nor a trained checkpoint.
"""

from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from config import LossConfig
from model import CabinSepLoss


class CabinSepLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        self.loss = CabinSepLoss(16_000)
        self.generator = torch.Generator().manual_seed(927)

    def signals(self, batch: int = 4, samples: int = 901):
        speech = 0.1 * torch.randn(batch, 4, samples, generator=self.generator)
        noise = 0.05 * torch.randn(batch, 4, samples, generator=self.generator)
        # Different numbers of active zones expose accidental active-zone
        # weighting of examples when the loss groups equal waveform lengths.
        for index in range(batch):
            speech[index, index % 4 + 1 :] = 0
        estimated_speech = speech + 0.03 * torch.randn(
            speech.shape, generator=self.generator
        )
        estimated_noise = noise + 0.02 * torch.randn(
            noise.shape, generator=self.generator
        )
        return estimated_speech, speech, estimated_noise, noise

    def test_variable_lengths_match_independently_cropped_examples(self) -> None:
        inputs = self.signals()
        lengths = torch.tensor([130, 513, 130, 901])
        actual = self.loss(*inputs, lengths=lengths)
        individually_cropped = [
            self.loss(*(value[index : index + 1, :, :stop] for value in inputs))
            for index, stop in enumerate(lengths.tolist())
        ]
        for key, value in actual.items():
            with self.subTest(component=key):
                expected = torch.stack([item[key] for item in individually_cropped]).mean()
                torch.testing.assert_close(value, expected, rtol=1e-5, atol=2e-6)

    def test_padding_has_no_effect_on_loss_or_gradients(self) -> None:
        inputs = tuple(value.clone() for value in self.signals())
        lengths = torch.tensor([1, 233, 513, 901])
        baseline = self.loss(*inputs, lengths=lengths)
        # Deliberately nonzero padding is essential: zero padding can hide a
        # missed mask in waveform means and the final overlapping FBank frame.
        for value in inputs:
            for index, stop in enumerate(lengths.tolist()):
                value[index, :, stop:] = 73.0 + index
        inputs[0].requires_grad_()
        inputs[2].requires_grad_()
        result = self.loss(*inputs, lengths=lengths)
        for key in baseline:
            torch.testing.assert_close(result[key], baseline[key], rtol=0, atol=0)
        result["loss"].backward()
        for estimate in (inputs[0], inputs[2]):
            self.assertIsNotNone(estimate.grad)
            self.assertTrue(bool(torch.isfinite(estimate.grad).all()))
            for index, stop in enumerate(lengths.tolist()):
                with self.subTest(estimate_shape=tuple(estimate.shape), item=index):
                    self.assertEqual(int(torch.count_nonzero(estimate.grad[index, :, stop:])), 0)
            self.assertGreater(float(estimate.grad.abs().sum()), 0.0)

    def test_active_silent_and_single_sample_examples_have_finite_backward(self) -> None:
        for pattern in ("active", "some_silent", "all_silent", "all_zero", "one_sample"):
            with self.subTest(pattern=pattern):
                samples = 1 if pattern == "one_sample" else 417
                estimated_speech, speech, estimated_noise, noise = self.signals(3, samples)
                if pattern == "active":
                    speech = 0.1 * torch.randn(speech.shape, generator=self.generator)
                elif pattern in {"all_silent", "all_zero", "one_sample"}:
                    speech.zero_()
                if pattern == "all_zero":
                    estimated_speech.zero_()
                    estimated_noise.zero_()
                    noise.zero_()
                estimated_speech.requires_grad_()
                estimated_noise.requires_grad_()
                result = self.loss(estimated_speech, speech, estimated_noise, noise)
                for value in result.values():
                    self.assertTrue(bool(torch.isfinite(value)))
                result["loss"].backward()
                for estimate in (estimated_speech, estimated_noise):
                    self.assertIsNotNone(estimate.grad)
                    self.assertTrue(bool(torch.isfinite(estimate.grad).all()))

    def test_fbank_supervises_first_and_final_samples_including_short_clips(self) -> None:
        for samples in (1, 20, 399, 400, 401, 511, 513, 901):
            for position in sorted({0, samples - 1}):
                with self.subTest(samples=samples, position=position):
                    estimate = torch.zeros(1, 1, samples)
                    estimate[0, 0, position] = 0.2
                    estimate.requires_grad_()
                    value = self.loss.fbank_mae(estimate, torch.zeros_like(estimate))
                    self.assertGreater(float(value), 0.0)
                    value.backward()
                    self.assertTrue(bool(torch.isfinite(estimate.grad).all()))
                    self.assertGreater(float(estimate.grad[0, 0, position].abs()), 0.0)

    def test_invalid_lengths_are_rejected(self) -> None:
        inputs = self.signals(batch=2, samples=257)
        invalid = (
            100,
            [100],
            [[100], [200]],
            [0, 200],
            [-1, 200],
            [100, 258],
            [True, True],
            [100.5, 200.0],
            [float("nan"), 200],
            [float("inf"), 200],
            torch.tensor([100 + 0j, 200 + 0j]),
        )
        for lengths in invalid:
            with self.subTest(lengths=lengths):
                with self.assertRaises(ValueError):
                    self.loss(*inputs, lengths=lengths)

    def test_invalid_waveform_shapes_are_rejected(self) -> None:
        for shapes in (((0, 4, 30),) * 4, ((2, 4, 0),) * 4, ((2, 30),) * 4,
                       ((2, 4, 30), (2, 4, 31), (2, 4, 30), (2, 4, 30)),
                       ((2, 4, 30), (2, 4, 30), (2, 3, 30), (2, 3, 30))):
            with self.subTest(shapes=shapes):
                with self.assertRaises(ValueError):
                    self.loss(*(torch.zeros(shape) for shape in shapes))

    def test_invalid_loss_configuration_is_rejected(self) -> None:
        invalid = (
            {"speech_fbank_weight": -1.0},
            {"speech_si_snr_weight": float("nan")},
            {"noise_fbank_weight": float("inf")},
            {"speech_fbank_weight": 0.0, "speech_si_snr_weight": 0.0,
             "noise_fbank_weight": 0.0},
            {"fbank_n_fft": 0},
            {"fbank_num_mels": 0},
            {"epsilon": 0.0},
            {"epsilon": -1.0},
            {"epsilon": float("inf")},
            {"epsilon": float("nan")},
            {"fbank_hop_ms": 0.0},
            {"fbank_hop_ms": 30.0},
            {"fbank_window_ms": 40.0},
            {"fbank_window_ms": float("nan")},
            {"fbank_window_ms": float("inf")},
            {"fbank_hop_ms": float("inf")},
            {"fbank_min_hz": -1.0},
            {"fbank_min_hz": 8_000.0},
            {"fbank_max_hz": 8_001.0},
            {"fbank_max_hz": 10.0},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    CabinSepLoss(16_000, replace(LossConfig(), **changes))
        for sample_rate in (0, -16_000):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    CabinSepLoss(sample_rate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
