"""Run-button CPU regressions for loader handoff, training and checkpoint resume."""

from dataclasses import asdict, replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import CabinSepConfig, LossConfig, TrainingRunConfig
from model import CabinSep
from train import _load_checkpoint, adapt_batch, forward_train_batch, run_training


def tiny_model_config():
    return CabinSepConfig.small(
        n_fft=64, window_ms=4.0, hop_ms=2.0,
        encoder_hidden_channels=2, encoder_output_channels=2,
        fusion_channels=8, fullband_conv_channels=2, fullband_lstm_hidden=8,
        subband_hidden=8, conformer_heads=2, conformer_layers=1,
        conformer_ffn_dim=4, mask_noise_hidden=8,
    )


def tiny_loss_config():
    return LossConfig(
        speech_fbank_weight=0.03, fbank_n_fft=64, fbank_window_ms=4.0,
        fbank_hop_ms=2.0, fbank_num_mels=8,
    )


class RandomAudioDataset(Dataset):
    """Exercise sampler-generator plus Python, NumPy and Torch RNG restoration."""

    def __len__(self):
        return 5

    def __getitem__(self, index):
        speech = torch.randn(4, 128) * (0.05 + random.random() * 0.01)
        noise = torch.from_numpy(np.random.randn(4, 128).astype(np.float32)) * 0.01
        return {"mixture": speech + noise, "speech": speech, "noise": noise,
                "lengths": 96 + index * 8, "sample_rate": 16000}


def shuffled_loader():
    return DataLoader(RandomAudioDataset(), batch_size=2, shuffle=True,
                      generator=torch.Generator().manual_seed(123), num_workers=0)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model_config = tiny_model_config()
        self.loss_config = tiny_loss_config()
        self.run_config = TrainingRunConfig(
            checkpoint_dir=self.temporary.name, run_name="test", device="cpu",
            epochs=2, max_steps=2, lr_halving_steps=2,
            log_every_steps=100, validate_every_steps=100, checkpoint_every_steps=100,
        )
        generator = torch.Generator().manual_seed(17)
        speech = torch.randn(2, 4, 128, generator=generator) * 0.1
        noise = torch.randn(2, 4, 128, generator=generator) * 0.01
        self.batch = {"mixture": speech + noise, "speech": speech, "noise": noise,
                      "lengths": torch.tensor([96, 128]), "sample_rate": 16000}

    def train(self, config=None, **kwargs):
        kwargs.setdefault("train_loader", [self.batch, self.batch, self.batch])
        if not (config or self.run_config).resume_checkpoint and (config or self.run_config).stage == 1:
            kwargs.setdefault("model_config", self.model_config)
            kwargs.setdefault("loss_config", self.loss_config)
        return run_training(config or self.run_config, **kwargs)

    def test_layout_lengths_and_padding(self):
        batch = {key: value.permute(0, 2, 1) if isinstance(value, torch.Tensor) and value.ndim == 3 else value
                 for key, value in self.batch.items()}
        mixture, speech, noise, lengths = adapt_batch(
            batch, replace(self.run_config, batch_layout="BLZ"), self.model_config, torch.device("cpu")
        )
        self.assertEqual(tuple(mixture.shape), (2, 4, 128))
        self.assertEqual(lengths.tolist(), [96, 128])
        for tensor in (mixture, speech, noise):
            self.assertTrue(torch.equal(tensor[0, :, 96:], torch.zeros(4, 32)))
        self.assertTrue(torch.equal(mixture[1], self.batch["mixture"][1]))
        self.assertFalse(torch.equal(self.batch["mixture"][0, :, 96:], torch.zeros(4, 32)))

    def test_real_model_padded_forward_matches_individual_lengths(self):
        model = CabinSep(self.model_config).eval()
        mixture, _, _, lengths = adapt_batch(
            self.batch, self.run_config, self.model_config, torch.device("cpu")
        )
        with torch.no_grad():
            grouped = forward_train_batch(model, mixture, lengths)
            first = model.forward_train(mixture[:1, :, :96])
            second = model.forward_train(mixture[1:, :, :128])
        for name in ("speech", "noise"):
            torch.testing.assert_close(grouped[name][:1, :, :96], first[name])
            torch.testing.assert_close(grouped[name][1:, :, :128], second[name])
            self.assertEqual(grouped[name][0, :, 96:].abs().sum().item(), 0.0)

    def test_invalid_batches_fail_clearly(self):
        invalid = [
            ({**self.batch, "lengths": [0, 128]}, ValueError),
            ({**self.batch, "lengths": [96.5, 128]}, ValueError),
            ({**self.batch, "lengths": [96]}, ValueError),
            ({**self.batch, "sample_rate": []}, ValueError),
            ({**self.batch, "sample_rate": 8000}, ValueError),
            ({**self.batch, "noise": self.batch["noise"].long()}, TypeError),
            ({key: value[:, :, :0] if isinstance(value, torch.Tensor) and value.ndim == 3 else value
              for key, value in self.batch.items()}, ValueError),
            ((self.batch["mixture"],), TypeError),
        ]
        for batch, error in invalid:
            with self.subTest(error=error, keys=str(type(batch))):
                with self.assertRaises(error):
                    adapt_batch(batch, self.run_config, self.model_config, torch.device("cpu"))

    def test_mid_epoch_resume_matches_uninterrupted_with_shuffling(self):
        uninterrupted_path = self.train(
            replace(self.run_config, run_name="uninterrupted", max_steps=6), train_loader=shuffled_loader()
        )
        partial_path = self.train(train_loader=shuffled_loader())
        partial = _load_checkpoint(partial_path)
        self.assertEqual((partial["next_epoch_index"], partial["next_batch_index"]), (0, 2))
        resumed_path = self.train(
            replace(self.run_config, resume_checkpoint=str(partial_path), max_steps=6),
            train_loader=shuffled_loader(),
        )
        uninterrupted, resumed = _load_checkpoint(uninterrupted_path), _load_checkpoint(resumed_path)
        self.assertEqual(resumed["global_step"], 6)
        self.assertEqual(resumed["loss_config"], asdict(self.loss_config))
        self.assertEqual(resumed["scheduler_state_dict"], uninterrupted["scheduler_state_dict"])
        for name, value in uninterrupted["model_state_dict"].items():
            self.assertTrue(torch.equal(value, resumed["model_state_dict"][name]), name)

    def test_resume_at_total_limit_does_not_update_or_require_loader(self):
        path = self.train()
        before = path.read_bytes()
        result = run_training(replace(self.run_config, resume_checkpoint=str(path)))
        self.assertEqual(result, path)
        self.assertEqual(before, path.read_bytes())

    def test_stage2_preserves_model_and_loss_with_fresh_optimizer(self):
        path = self.train()
        stage2 = self.train(replace(
            self.run_config, stage=2, stage1_checkpoint=str(path), run_name="stage2",
            max_steps=1, learning_rate=0.0003,
        ))
        checkpoint = _load_checkpoint(stage2)
        self.assertEqual(checkpoint["stage"], 2)
        self.assertEqual(checkpoint["global_step"], 1)
        self.assertEqual(checkpoint["model_config"], asdict(self.model_config))
        self.assertEqual(checkpoint["loss_config"], asdict(self.loss_config))
        self.assertEqual(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"], 0.0003)
        self.assertTrue((stage2.parent / "best.pt").is_file())

    def test_best_selection_and_latest_validation_metadata(self):
        config = replace(self.run_config, validate_every_steps=1, checkpoint_every_steps=1)
        with patch("train.validate", side_effect=[{"loss": 1.0}, {"loss": 2.0}]) as validate_mock:
            path = self.train(config, validation_loader=[self.batch])
        self.assertEqual(validate_mock.call_count, 2)
        latest = _load_checkpoint(path)
        best = _load_checkpoint(path.parent / "best.pt")
        periodic = _load_checkpoint(path.parent / "step_00000001.pt")
        self.assertEqual((best["global_step"], best["best_validation_loss"]), (1, 1.0))
        self.assertEqual(latest["best_validation_loss"], 1.0)
        self.assertEqual(periodic["best_validation_loss"], 1.0)

    def test_empty_loaders_and_one_shot_iterators(self):
        for index, (loader, validation, error, message) in enumerate([
            ([], None, ValueError, "Training loader yielded no batches"),
            ([self.batch], [], ValueError, "Validation loader yielded no batches"),
            (iter([self.batch]), None, TypeError, "reusable"),
        ]):
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    self.train(replace(self.run_config, run_name=f"empty_{index}"),
                               train_loader=loader, validation_loader=validation)

    def test_invalid_optimizer_values_and_checkpoint_config_conflicts(self):
        for values in ({"learning_rate": float("nan")}, {"weight_decay": -1},
                       {"gradient_clip_norm": 0}, {"epochs": 1.5}, {"run_name": ".."}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.train(replace(self.run_config, **values))
        path = self.train()
        with self.assertRaisesRegex(ValueError, "loss_config differs"):
            self.train(replace(self.run_config, resume_checkpoint=str(path), max_steps=3),
                       loss_config=LossConfig())


if __name__ == "__main__":
    unittest.main(verbosity=2)
