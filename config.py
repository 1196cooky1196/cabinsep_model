"""Central, editable configuration for CabinSep.

The project intentionally does not require command-line arguments.  Edit the
``TRAIN_RUN`` or ``INFERENCE_RUN`` objects at the bottom of this file and press
Run in the corresponding script.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class CabinSepConfig:
    """CabinSep architecture and signal-processing configuration.

    Values explicitly reported in the CabinSep paper are marked ``[PAPER]``.
    Values inherited from FSB-LSTM (CabinSep reference [30]) are marked
    ``[REFERENCE]``.  Unreported implementation choices are marked
    ``[ASSUMPTION]`` and summarized in IMPLEMENTATION_NOTES.md.
    """

    variant: str = "S"
    num_zones: int = 4  # [PAPER] Experiments use four microphone zones.
    sample_rate: int = 16_000  # [ASSUMPTION] The paper does not state it.
    n_fft: int = 512  # [PAPER]
    window_ms: float = 32.0  # [PAPER]
    hop_ms: float = 16.0  # [PAPER]
    stft_center: bool = False  # [ASSUMPTION] Required for causal operation.
    feature_epsilon: float = 1.0e-8
    front_ipd_pair: Tuple[int, int] = (0, 1)  # Paper zones 1/2 in zero-based Python.

    # [ASSUMPTION] Each paper encoder is Conv-ReLU-Conv-ReLU, but its
    # dimensions and kernels are not reported.
    encoder_hidden_channels: int = 8
    encoder_output_channels: int = 8
    encoder_kernel: Tuple[int, int] = (1, 3)

    fusion_channels: int = 24  # [PAPER] C = 24.
    fusion_kernel: Tuple[int, int] = (1, 3)  # [REFERENCE]/[ASSUMPTION]

    num_full_sub_modules: int = 1
    tac_compression_ratio: int = 4
    # [DERIVED] Eq. (6) says channel, but Table 1's 0.40 M TAC ablation can
    # only be reconciled by applying the Linear layers on F.  Keep "channel"
    # available for the literal equation interpretation.
    tac_axis: str = "frequency"
    time_skip: bool = False  # [PAPER] Table 1 base; True enables +time skip ablation.
    inference_time_skip_offset: int = 0  # [ASSUMPTION] Deterministic parity.

    # Full-band block.  CabinSep delegates its structure to reference [30].
    fullband_conv_channels: int = 8  # [REFERENCE]
    fullband_frequency_kernel: int = 8  # [REFERENCE]
    fullband_frequency_stride: int = 4  # [REFERENCE]
    fullband_lstm_hidden: int = 256  # [REFERENCE]
    fullband_lstm_layers: int = 1  # [REFERENCE]

    # Sub-band Conformer.
    subband_hidden: int = 16  # [PAPER] H = 16.
    subband_frequency_kernel: int = 5  # [REFERENCE]
    subband_frequency_stride: int = 5  # [REFERENCE]
    conformer_ffn_dim: int = 8  # [PAPER] H / 2 = 8.
    conformer_heads: int = 4  # [PAPER]
    conformer_relative_position: bool = True  # [REFERENCE] Conformer [32].
    conformer_layers: int = 4
    conformer_conv_kernel: int = 32  # [REFERENCE] Original Conformer [32].
    conformer_conv_norm: str = "batch"  # [REFERENCE] Original Conformer [32].
    conformer_dropout: float = 0.1  # [REFERENCE] Original Conformer [32].
    # None reproduces the paper's base model.  Set e.g. 125 at 16 ms hops for
    # the paper's optional ~2 s look-back chunk ablation.
    conformer_left_context_frames: Optional[int] = None

    # [ASSUMPTION] Figure 1(e) omits dimensions. The Table 1 noise-head
    # ablation is consistent with F as the Linear axis but does not prove it.
    # None uses F hidden units after GLU (Linear F -> 2F -> GLU -> F -> F).
    mask_noise_hidden: Optional[int] = None

    # Inference-only online covariance update.
    mvdr_forgetting_factor: float = 1.0
    mvdr_diagonal_loading: float = 1.0e-4
    mvdr_epsilon: float = 1.0e-8

    @property
    def window_length(self) -> int:
        return int(round(self.sample_rate * self.window_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def num_frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    def validate(self) -> "CabinSepConfig":
        variant = self.variant.upper()
        if variant not in {"S", "M", "L"}:
            raise ValueError(f"variant must be S, M, or L; got {self.variant!r}")
        if self.sample_rate <= 0 or self.n_fft <= 0 or self.n_fft % 2:
            raise ValueError("sample_rate and an even n_fft must be positive")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (self.window_ms, self.hop_ms)
        ):
            raise ValueError("STFT window_ms and hop_ms must be finite and positive")
        if self.num_zones < 2:
            raise ValueError("CabinSep IPD extraction requires at least two zones")
        if len(self.front_ipd_pair) != 2 or self.front_ipd_pair[0] == self.front_ipd_pair[1]:
            raise ValueError("front_ipd_pair must contain two distinct zone indices")
        if max(self.front_ipd_pair) >= self.num_zones or min(self.front_ipd_pair) < 0:
            raise ValueError("front_ipd_pair contains an invalid zero-based zone index")
        if self.stft_center:
            raise ValueError("CabinSep is causal; stft_center must remain False")
        if self.n_fft != self.window_length:
            raise ValueError(
                "This implementation requires n_fft == 32 ms window length for "
                "causal, exactly invertible center=False STFT. Adjust both n_fft "
                "and sample_rate together."
            )
        if not 0 < self.hop_length <= self.window_length:
            raise ValueError("hop length must be in (0, window length]")
        if not math.isfinite(self.feature_epsilon) or self.feature_epsilon <= 0.0:
            raise ValueError("feature_epsilon must be finite and positive")
        if self.encoder_kernel[0] != 1 or self.fusion_kernel[0] != 1:
            raise ValueError("Causal CabinSep requires time-axis convolution kernels of size 1")
        if (
            self.encoder_kernel[1] <= 0
            or self.fusion_kernel[1] <= 0
            or self.encoder_kernel[1] % 2 == 0
            or self.fusion_kernel[1] % 2 == 0
        ):
            raise ValueError("Same-size encoder/fusion frequency kernels must be positive and odd")
        if self.tac_compression_ratio <= 0:
            raise ValueError("TAC compression ratio must be positive")
        if self.tac_axis not in {"channel", "frequency"}:
            raise ValueError("tac_axis must be channel or frequency (legacy)")
        if not isinstance(self.time_skip, bool):
            raise ValueError("time_skip must be boolean")
        if self.tac_axis == "channel" and self.fusion_channels % self.tac_compression_ratio:
            raise ValueError(
                "fusion_channels must be divisible by TAC compression ratio for channel TAC"
            )
        if min(
            self.encoder_hidden_channels,
            self.encoder_output_channels,
            self.fusion_channels,
            self.fullband_conv_channels,
            self.fullband_frequency_kernel,
            self.fullband_frequency_stride,
            self.fullband_lstm_hidden,
            self.fullband_lstm_layers,
            self.subband_hidden,
            self.subband_frequency_kernel,
            self.subband_frequency_stride,
            self.conformer_ffn_dim,
            self.conformer_heads,
        ) <= 0:
            raise ValueError("All neural dimensions, kernels, strides, and layer counts must be positive")
        if self.num_full_sub_modules <= 0 or self.conformer_layers <= 0:
            raise ValueError("Full-Sub module and Conformer layer counts must be positive")
        if self.inference_time_skip_offset not in (0, 1):
            raise ValueError("inference_time_skip_offset must be 0 or 1")
        if self.subband_hidden % self.conformer_heads:
            raise ValueError("subband_hidden must be divisible by conformer_heads")
        if not isinstance(self.conformer_relative_position, bool):
            raise ValueError("conformer_relative_position must be boolean")
        if self.conformer_conv_kernel < 1:
            raise ValueError("conformer_conv_kernel must be positive")
        if self.conformer_conv_norm not in {"batch", "layer"}:
            raise ValueError("conformer_conv_norm must be batch or layer (legacy)")
        if (
            self.conformer_left_context_frames is not None
            and self.conformer_left_context_frames <= 0
        ):
            raise ValueError("conformer_left_context_frames must be positive when set")
        if self.mask_noise_hidden is not None and self.mask_noise_hidden <= 0:
            raise ValueError("mask_noise_hidden must be positive when set")
        if not 0.0 <= self.conformer_dropout < 1.0:
            raise ValueError("conformer_dropout must be in [0, 1)")
        if (
            not math.isfinite(self.mvdr_forgetting_factor)
            or not 0.0 < self.mvdr_forgetting_factor <= 1.0
        ):
            raise ValueError("mvdr_forgetting_factor must be in (0, 1]")
        if (
            not math.isfinite(self.mvdr_diagonal_loading)
            or not math.isfinite(self.mvdr_epsilon)
            or self.mvdr_diagonal_loading < 0.0
            or self.mvdr_epsilon <= 0.0
        ):
            raise ValueError("MVDR diagonal loading must be nonnegative and epsilon positive")
        return self

    @classmethod
    def small(cls, **overrides: object) -> "CabinSepConfig":
        config = cls(
            variant="S",
            num_full_sub_modules=1,
            tac_compression_ratio=4,
            conformer_layers=4,
        )
        return replace(config, **overrides).validate()

    @classmethod
    def medium(cls, **overrides: object) -> "CabinSepConfig":
        config = cls(
            variant="M",
            num_full_sub_modules=2,
            tac_compression_ratio=4,
            conformer_layers=2,
        )
        return replace(config, **overrides).validate()

    @classmethod
    def large(cls, **overrides: object) -> "CabinSepConfig":
        config = cls(
            variant="L",
            num_full_sub_modules=3,
            tac_compression_ratio=2,
            conformer_layers=2,
        )
        return replace(config, **overrides).validate()

    @classmethod
    def from_variant(cls, variant: str, **overrides: object) -> "CabinSepConfig":
        factories = {"S": cls.small, "M": cls.medium, "L": cls.large}
        key = variant.upper()
        if key not in factories:
            raise ValueError(f"Unknown CabinSep variant: {variant!r}")
        return factories[key](**overrides)


@dataclass(frozen=True)
class LossConfig:
    # [PAPER] Equation (7).
    speech_fbank_weight: float = 0.01
    speech_si_snr_weight: float = 1.0
    noise_fbank_weight: float = 0.01

    # [ASSUMPTION] CabinSep does not report its FBank extraction settings.
    fbank_num_mels: int = 80
    fbank_n_fft: int = 512
    fbank_window_ms: float = 25.0
    fbank_hop_ms: float = 10.0
    fbank_min_hz: float = 20.0
    fbank_max_hz: Optional[float] = None
    epsilon: float = 1.0e-8


@dataclass(frozen=True)
class TrainingRunConfig:
    """Edit this object and press Run on train.py.

    The external team should expose the function named by ``loader_factory``.
    Its precise, deliberately small contract is documented in train.py.
    """

    model_variant: str = "S"
    loader_factory: str = "data_bridge:build_dataloaders"
    stage: int = 1
    stage1_checkpoint: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    run_name: str = "cabinsep_s_stage1"
    device: str = "auto"
    seed: int = 42
    epochs: int = 100
    max_steps: Optional[int] = None
    learning_rate: float = 1.0e-4  # [PAPER]
    lr_halving_steps: int = 20_000  # [PAPER]
    weight_decay: float = 0.0
    gradient_clip_norm: Optional[float] = None
    log_every_steps: int = 20
    validate_every_steps: int = 1_000
    checkpoint_every_steps: int = 1_000
    mixture_key: str = "mixture"
    speech_key: str = "speech"
    noise_key: str = "noise"
    lengths_key: Optional[str] = "lengths"  # Optional valid sample counts [B].
    batch_layout: str = "BZL"  # Set to BLZ only if the delivered loader uses it.


@dataclass(frozen=True)
class InferenceRunConfig:
    """Edit this object and press Run on inference.py."""

    checkpoint_path: str = "checkpoints/cabinsep_s_stage1/best.pt"
    input_path: str = "input/multichannel.wav"
    output_dir: str = "outputs/inference"
    device: str = "auto"
    fallback_model_variant: str = "S"
    save_masks: bool = True


# ---------------------------------------------------------------------------
# Run-button settings. No terminal arguments are required.
# ---------------------------------------------------------------------------

# Choose small(), medium(), or large() and put any architecture choices here.
# Resume/Stage-2 runs always restore the architecture and loss from checkpoint.
TRAIN_MODEL = CabinSepConfig.small()
TRAIN_LOSS = LossConfig()
TRAIN_RUN = TrainingRunConfig(model_variant=TRAIN_MODEL.variant)
INFERENCE_RUN = InferenceRunConfig()
