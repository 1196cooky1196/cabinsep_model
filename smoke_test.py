"""Run-button smoke test for the complete CabinSep path.

No dataset, checkpoint, command-line argument, or GPU is required.  Open this
file in the IDE and press Run.
"""

from __future__ import annotations

import random

import torch

from config import CabinSepConfig, LossConfig
from model import CabinSep, CabinSepLoss


SEED = 7
BATCH_SIZE = 2
WAVEFORM_SAMPLES = 2_048


def _shape(value: torch.Tensor) -> str:
    return str(list(value.shape))


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    config = CabinSepConfig.small()
    model = CabinSep(config)
    criterion = CabinSepLoss(config.sample_rate, LossConfig())

    # Synthetic zone-aligned targets; this is only a graph/shape test.
    target_speech = 0.1 * torch.randn(BATCH_SIZE, config.num_zones, WAVEFORM_SAMPLES)
    target_noise = 0.03 * torch.randn_like(target_speech)
    mixture = target_speech + target_noise

    print("CabinSep smoke test")
    print(f"variant                 : CabinSep-{config.variant}")
    print(f"input waveform          : {_shape(mixture)}")

    model.train()
    estimates = model.estimate_masks(mixture, return_intermediates=True)
    spectrum = estimates["spectrum"]
    speech_mask = estimates["speech_mask"]
    noise_mask = estimates["noise_mask"]
    intermediates = estimates["intermediates"]
    assert isinstance(spectrum, torch.Tensor)
    assert isinstance(speech_mask, torch.Tensor)
    assert isinstance(noise_mask, torch.Tensor)
    assert isinstance(intermediates, dict)

    print(f"STFT Y                 : {_shape(spectrum)} complex={spectrum.is_complex()}")
    print(f"Spec [Re, Im]          : {_shape(intermediates['spec'])}")
    print(f"LPS                    : {_shape(intermediates['lps'])}")
    print(f"IPD [cos, sin]         : {_shape(intermediates['ipd'])}")
    print(f"Spec encoder           : {_shape(intermediates['spec_embedding'])}")
    print(f"LPS encoder            : {_shape(intermediates['lps_embedding'])}")
    print(f"IPD encoder            : {_shape(intermediates['ipd_embedding'])}")
    print(f"Fusion Conv2d          : {_shape(intermediates['fused_embedding'])}")
    for index in range(config.num_full_sub_modules):
        print(
            f"Full-Sub module {index + 1:<2}    : "
            f"{_shape(intermediates[f'full_sub_{index + 1}'])}"
        )
    print(f"Decoded mask features  : {_shape(intermediates['decoded_mask_features'])}")
    print(f"speech mask            : {_shape(speech_mask)}")
    print(f"noise mask             : {_shape(noise_mask)}")

    expected_frames = 1 + (WAVEFORM_SAMPLES - config.n_fft) // config.hop_length
    expected_mask_shape = (
        BATCH_SIZE,
        config.num_zones,
        expected_frames,
        config.num_frequency_bins,
    )
    assert tuple(spectrum.shape) == expected_mask_shape
    assert tuple(speech_mask.shape) == expected_mask_shape
    assert tuple(noise_mask.shape) == expected_mask_shape
    assert bool(((speech_mask >= 0.0) & (speech_mask <= 1.0)).all())
    assert bool((noise_mask >= 0.0).all())

    # STFT/iSTFT should preserve the unmodified waveform.
    round_trip = model.istft(spectrum, WAVEFORM_SAMPLES)
    round_trip_error = (round_trip - mixture).abs().max().item()
    assert round_trip_error < 1.0e-4, round_trip_error

    # Complete differentiable mask-training path and Paper Eq. (7).
    training_output = model.forward_train(mixture)
    losses = criterion(
        training_output["speech"],
        target_speech,
        training_output["noise"],
        target_noise,
    )
    assert bool(torch.isfinite(losses["loss"]))
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    print(f"training waveform      : {_shape(training_output['speech'])}")
    print(f"total loss             : {losses['loss'].item():.6f}")
    print(f"STFT round-trip max err: {round_trip_error:.3e}")

    # Prefix causality: adding future samples must not change existing masks.
    model.eval()
    with torch.no_grad():
        full_eval = model.estimate_masks(mixture[:1])
        prefix_samples = 1_536  # hop-aligned: exactly five STFT frames.
        prefix_eval = model.estimate_masks(mixture[:1, :, :prefix_samples])
        prefix_frames = prefix_eval["speech_mask"].shape[2]
        causal_difference = (
            full_eval["speech_mask"][:, :, :prefix_frames]
            - prefix_eval["speech_mask"]
        ).abs().max().item()
        assert causal_difference < 1.0e-5, causal_difference

        # Complete inference path. MVDR is intentionally evaluated only here.
        inference_output = model.separate(mixture[:1])
    separated = inference_output["waveform"]
    assert isinstance(separated, torch.Tensor)
    assert tuple(separated.shape) == (1, config.num_zones, WAVEFORM_SAMPLES)
    assert bool(torch.isfinite(separated).all())
    print(f"MVDR+iSTFT waveform    : {_shape(separated)}")
    print(f"prefix causality max err: {causal_difference:.3e}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"parameters             : {parameter_count:,}")
    print(f"trainable parameters   : {trainable_count:,}")
    print("PASS: model, loss, backward, and inference paths are finite and shape-correct.")


if __name__ == "__main__":
    main()
