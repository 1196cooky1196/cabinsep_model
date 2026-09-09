"""Readable CabinSep baseline implementation in PyTorch.

Canonical layouts used throughout this file:

* waveform: ``[B, Z, L]``
* complex spectrum: ``[B, Z, T, F]``
* real feature map: ``[B, C, T, F]``

The CabinSep paper leaves several implementation details unspecified.  Every
such choice is tagged [ASSUMPTION], [DERIVED], or [REFERENCE] in this file and
summarized in IMPLEMENTATION_NOTES.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from config import CabinSepConfig, LossConfig


# ============================================================
# Common reusable blocks and tensor-layout helpers
# ============================================================


def _assert_feature_map(x: Tensor, channels: Optional[int] = None) -> None:
    """Validate a real feature map with canonical layout [B, C, T, F]."""

    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, T, F], got shape {tuple(x.shape)}")
    if channels is not None and x.shape[1] != channels:
        raise ValueError(f"Expected C={channels}, got C={x.shape[1]}")


def _right_padded_size(size: int, kernel: int, stride: int) -> int:
    """Smallest Q >= size for which (Q-kernel) is divisible by stride."""

    if kernel <= 0 or stride <= 0:
        raise ValueError("kernel and stride must be positive")
    steps = max(0, math.ceil((size - kernel) / stride))
    return steps * stride + kernel


def to_subband_sequences(x: Tensor) -> Tuple[Tensor, Tuple[int, int, int, int]]:
    """Convert [B, H, T, F] to F temporal sequences [B*F, T, H]."""

    _assert_feature_map(x)
    batch, hidden, frames, bands = x.shape
    sequences = x.permute(0, 3, 2, 1).contiguous().reshape(batch * bands, frames, hidden)
    return sequences, (batch, hidden, frames, bands)


def from_subband_sequences(x: Tensor, shape: Tuple[int, int, int, int]) -> Tensor:
    """Restore [B*F, T, H] to [B, H, T, F]."""

    batch, hidden, frames, bands = shape
    expected = (batch * bands, frames, hidden)
    if tuple(x.shape) != expected:
        raise ValueError(f"Expected sub-band sequence shape {expected}, got {tuple(x.shape)}")
    return x.reshape(batch, bands, frames, hidden).permute(0, 3, 2, 1).contiguous()


class ConvAct(nn.Module):
    """A repeated Conv2d + ReLU unit used by the three feature encoders."""

    def __init__(self, in_channels: int, out_channels: int, kernel: Tuple[int, int]) -> None:
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv(x))


class CausalGlobalLayerNormSequence(nn.Module):
    """Reference [30] causal global normalization for [B, T, A]."""

    def __init__(self, features: int, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        self.features = features
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.features:
            raise ValueError(f"Expected [B, T, {self.features}], got {tuple(x.shape)}")
        # Half-precision cumulative counts overflow after only a few seconds.
        # Keep statistics in float32 under mixed precision, as LayerNorm does.
        statistics = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        running_sum = statistics.sum(dim=-1).cumsum(dim=1)
        running_square_sum = statistics.square().sum(dim=-1).cumsum(dim=1)
        counts = (
            torch.arange(1, x.shape[1] + 1, device=x.device, dtype=statistics.dtype)
            * self.features
        )
        mean = running_sum / counts.unsqueeze(0)
        variance = running_square_sum / counts.unsqueeze(0) - mean.square()
        normalized = (statistics - mean.unsqueeze(-1)) * torch.rsqrt(
            variance.clamp_min(0.0).unsqueeze(-1) + self.epsilon
        )
        return (normalized * self.weight + self.bias).to(dtype=x.dtype)


class CausalGlobalLayerNormMap(nn.Module):
    """Reference [30] causal global normalization for [B, C, T, F]."""

    def __init__(self, channels: int, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        self.channels = channels
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        _assert_feature_map(x, self.channels)
        values_per_frame = x.shape[1] * x.shape[3]
        statistics = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        running_sum = statistics.sum(dim=(1, 3)).cumsum(dim=1)
        running_square_sum = statistics.square().sum(dim=(1, 3)).cumsum(dim=1)
        counts = (
            torch.arange(1, x.shape[2] + 1, device=x.device, dtype=statistics.dtype)
            * values_per_frame
        )
        mean = running_sum / counts.unsqueeze(0)
        variance = running_square_sum / counts.unsqueeze(0) - mean.square()
        normalized = (statistics - mean[:, None, :, None]) * torch.rsqrt(
            variance.clamp_min(0.0)[:, None, :, None] + self.epsilon
        )
        return (normalized * self.weight + self.bias).to(dtype=x.dtype)


# ============================================================
# Feature extraction and encoders
# ============================================================


class SpecEncoder(nn.Module):
    """Paper Sec. 3.2: two Conv-ReLU layers for stacked real/imaginary spectra."""

    def __init__(self, config: CabinSepConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ConvAct(2 * config.num_zones, config.encoder_hidden_channels, config.encoder_kernel),
            ConvAct(
                config.encoder_hidden_channels,
                config.encoder_output_channels,
                config.encoder_kernel,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input : [B, 2Z, T, F]
        # Output: [B, encoder_output_channels, T, F]
        _assert_feature_map(x)
        return self.network(x)


class LPSEncoder(nn.Module):
    """Paper Sec. 3.2: two Conv-ReLU layers for log-power spectra."""

    def __init__(self, config: CabinSepConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ConvAct(config.num_zones, config.encoder_hidden_channels, config.encoder_kernel),
            ConvAct(
                config.encoder_hidden_channels,
                config.encoder_output_channels,
                config.encoder_kernel,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input : [B, Z, T, F]
        # Output: [B, encoder_output_channels, T, F]
        _assert_feature_map(x)
        return self.network(x)


class IPDEncoder(nn.Module):
    """Paper Sec. 3.2: two Conv-ReLU layers for front-row IPD features."""

    def __init__(self, config: CabinSepConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ConvAct(2, config.encoder_hidden_channels, config.encoder_kernel),
            ConvAct(
                config.encoder_hidden_channels,
                config.encoder_output_channels,
                config.encoder_kernel,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input : [B, 2, T, F]
        # Output: [B, encoder_output_channels, T, F]
        _assert_feature_map(x, 2)
        return self.network(x)


# ============================================================
# Full-band LSTM (CabinSep reference [30])
# ============================================================


class FullBandLSTM(nn.Module):
    """Causal full-band block following FSB-LSTM reference [30].

    The frequency-downsampled channels and bands are flattened into one
    frame-level vector.  A unidirectional LSTM therefore sees all frequencies
    at a time step without looking into future frames.
    """

    def __init__(self, config: CabinSepConfig) -> None:
        super().__init__()
        channels = config.fusion_channels
        conv_channels = config.fullband_conv_channels
        kernel = config.fullband_frequency_kernel
        stride = config.fullband_frequency_stride
        frequency_bins = config.num_frequency_bins
        padded_bins = _right_padded_size(frequency_bins, kernel, stride)
        downsampled_bins = (padded_bins - kernel) // stride + 1
        frame_features = conv_channels * downsampled_bins

        self.channels = channels
        self.frequency_bins = frequency_bins
        self.padded_bins = padded_bins
        self.conv_channels = conv_channels
        self.downsampled_bins = downsampled_bins
        self.frame_features = frame_features

        self.downsample = nn.Conv2d(
            channels,
            conv_channels,
            kernel_size=(1, kernel),
            stride=(1, stride),
        )
        self.input_prelu = nn.PReLU(frame_features)
        self.input_norm = CausalGlobalLayerNormSequence(frame_features)
        self.lstm = nn.LSTM(
            input_size=frame_features,
            hidden_size=config.fullband_lstm_hidden,
            num_layers=config.fullband_lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.projection = nn.Linear(config.fullband_lstm_hidden, frame_features)
        self.output_norm = CausalGlobalLayerNormSequence(frame_features)
        self.output_prelu = nn.PReLU(frame_features)
        self.upsample = nn.ConvTranspose2d(
            conv_channels,
            channels,
            kernel_size=(1, kernel),
            stride=(1, stride),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input/output: [B, C, T, F]
        _assert_feature_map(x, self.channels)
        if x.shape[-1] != self.frequency_bins:
            raise ValueError(
                f"FullBandLSTM was built for F={self.frequency_bins}, got F={x.shape[-1]}"
            )
        residual = x
        x = functional.pad(x, (0, self.padded_bins - self.frequency_bins, 0, 0))
        x = self.downsample(x)
        batch, channels, frames, bands = x.shape
        if channels != self.conv_channels or bands != self.downsampled_bins:
            raise RuntimeError("Unexpected full-band downsampling shape")

        # [B, E, T, F_down] -> [B, A=E*F_down, T]
        x = x.permute(0, 1, 3, 2).contiguous().reshape(batch, self.frame_features, frames)
        x = self.input_prelu(x)
        # [B, A, T] -> temporal sequence [B, T, A]
        x = self.input_norm(x.transpose(1, 2))
        x, _ = self.lstm(x)
        x = self.projection(x)
        x = self.output_norm(x)
        x = self.output_prelu(x.transpose(1, 2)).transpose(1, 2)

        # [B, T, A] -> [B, E, T, F_down]
        x = x.reshape(batch, frames, self.conv_channels, self.downsampled_bins)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.upsample(x)[..., : self.frequency_bins]
        return residual + x


# ============================================================
# Time-skip TAC
# ============================================================


class TACBlock(nn.Module):
    """Paper Fig. 1(c), Eq. (6): transform and compress the channel axis.

    ``axis='frequency'`` is the reproduction default because it accounts for
    the ~0.40 M TAC parameters in Table 1.  ``axis='channel'`` follows the
    literal C -> C/d mapping in Eq. (6).  The paper contradicts itself here,
    so both interpretations stay explicit rather than hiding the choice.
    """

    def __init__(
        self, channels: int, frequency_bins: int, compression_ratio: int,
        axis: str = "frequency",
    ) -> None:
        super().__init__()
        if axis not in {"channel", "frequency"}:
            raise ValueError("TAC axis must be 'channel' or 'frequency'")
        if min(channels, frequency_bins, compression_ratio) <= 0:
            raise ValueError("TAC dimensions and compression ratio must be positive")
        if axis == "channel" and channels % compression_ratio:
            raise ValueError("TAC channel count must be divisible by compression ratio")
        self.axis = axis
        self.channels = channels
        self.frequency_bins = frequency_bins
        input_features = channels if axis == "channel" else frequency_bins
        self.compressed_features = max(1, input_features // compression_ratio)
        self.linear_a = nn.Linear(input_features, self.compressed_features)
        self.linear_b = nn.Linear(input_features, self.compressed_features)
        self.linear_c = nn.Linear(2 * self.compressed_features, input_features)

    def forward(self, x: Tensor) -> Tensor:
        # Input/output: [B, C, T_selected, F]
        _assert_feature_map(x, self.channels)
        if x.shape[-1] != self.frequency_bins:
            raise ValueError(f"TAC expected F={self.frequency_bins}, got {x.shape[-1]}")

        # nn.Linear acts on the last dimension; move C there for Eq. (6).
        transformed = x.permute(0, 2, 3, 1) if self.axis == "channel" else x
        branch_a = functional.relu(self.linear_a(transformed))
        branch_b = functional.relu(self.linear_b(transformed))
        mean_axis = -1 if self.axis == "channel" else 1
        channel_mean = branch_b.mean(dim=mean_axis, keepdim=True)
        channel_context = channel_mean.expand_as(branch_b)
        combined = torch.cat((branch_a, channel_context), dim=-1)
        output = functional.relu(self.linear_c(combined))
        return output.permute(0, 3, 1, 2).contiguous() if self.axis == "channel" else output


# ============================================================
# Sub-band Conformer
# ============================================================


class ConformerFeedForward(nn.Module):
    def __init__(self, hidden: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.network = nn.Sequential(
            nn.Linear(hidden, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(self.norm(x))


class CausalConformerConv(nn.Module):
    """Small causal Conformer convolution module for [N, T, H]."""

    def __init__(
        self, hidden: int, kernel_size: int, dropout: float, norm: str = "batch"
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(hidden)
        self.pointwise_in = nn.Conv1d(hidden, 2 * hidden, kernel_size=1)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            groups=hidden,
        )
        if norm == "batch":
            # [REFERENCE] Conformer [32], Fig. 2 / Sec. 2.2. At evaluation the
            # stored statistics are fixed, so this does not read future frames.
            self.batch_norm: Optional[nn.BatchNorm1d] = nn.BatchNorm1d(hidden)
            self.depthwise_norm: Optional[nn.LayerNorm] = None
        elif norm == "layer":
            # Compatibility option for checkpoints made by the earlier causal
            # baseline, which normalized every frame independently.
            self.batch_norm = None
            self.depthwise_norm = nn.LayerNorm(hidden)
        else:
            raise ValueError("Conformer convolution norm must be batch or layer")
        self.pointwise_out = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x).transpose(1, 2)
        x = functional.glu(self.pointwise_in(x), dim=1)
        # Left-only padding preserves causality.
        x = functional.pad(x, (self.kernel_size - 1, 0))
        x = self.depthwise(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        elif self.depthwise_norm is not None:
            x = self.depthwise_norm(x.transpose(1, 2)).transpose(1, 2)
        else:  # pragma: no cover - constructor guarantees one normalization.
            raise RuntimeError("Conformer convolution normalization is missing")
        x = functional.silu(x)
        x = self.dropout(self.pointwise_out(x))
        return x.transpose(1, 2)


class CausalConformerBlock(nn.Module):
    """Macaron-style causal Conformer block used inside each sub-band."""

    def __init__(
        self,
        hidden: int,
        ffn_dim: int,
        heads: int,
        conv_kernel: int,
        dropout: float,
        left_context_frames: Optional[int],
        relative_position: bool = True,
        convolution_norm: str = "batch",
    ) -> None:
        super().__init__()
        self.left_context_frames = left_context_frames
        self.relative_position = relative_position
        self.ffn1 = ConformerFeedForward(hidden, ffn_dim, dropout)
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        if relative_position:
            # [REFERENCE] Conformer [32], Sec. 2.1 uses Transformer-XL's
            # sinusoidal relative attention, including global content/position
            # biases. Keep the usual MHA projection keys for legacy models.
            self.relative_projection = nn.Linear(hidden, hidden, bias=False)
            self.content_bias = nn.Parameter(torch.zeros(heads, hidden // heads))
            self.position_bias = nn.Parameter(torch.zeros(heads, hidden // heads))
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution = CausalConformerConv(
            hidden, conv_kernel, dropout, norm=convolution_norm
        )
        self.ffn2 = ConformerFeedForward(hidden, ffn_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden)

    def _attention_mask(self, frames: int, device: torch.device) -> Tensor:
        query = torch.arange(frames, device=device)[:, None]
        key = torch.arange(frames, device=device)[None, :]
        allowed = key <= query
        if self.left_context_frames is not None:
            oldest = query - self.left_context_frames + 1
            allowed = allowed & (key >= oldest)
        return ~allowed  # MultiheadAttention: True entries are blocked.

    def _relative_attention(self, x: Tensor, mask: Tensor) -> Tensor:
        """Transformer-XL logits (Q+u)K^T + (Q+v)R_(query-key)^T.

        Only nonnegative lags are needed because future keys are masked. This
        is full-sequence causal attention; it does not maintain a streaming KV
        cache. Position vectors depend on relative lag, never sequence length.
        """
        batch, frames, hidden = x.shape
        heads = self.attention.num_heads
        head_dim = hidden // heads
        projected = functional.linear(
            x, self.attention.in_proj_weight, self.attention.in_proj_bias
        )
        query, key, value = (
            part.reshape(batch, frames, heads, head_dim).transpose(1, 2)
            for part in projected.chunk(3, dim=-1)
        )
        lag_count = min(frames, self.left_context_frames or frames)
        # Calculate sin/cos in float32 even when neural projections use AMP.
        positions = torch.arange(lag_count, device=x.device, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, hidden, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10_000.0) / hidden)
        )
        angles = positions[:, None] * frequencies[None, :]
        encoding = torch.zeros(lag_count, hidden, device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, : hidden // 2].cos()
        relative = self.relative_projection(encoding.to(dtype=x.dtype))
        relative = relative.reshape(lag_count, heads, head_dim).permute(1, 0, 2)

        content_logits = torch.matmul(
            query + self.content_bias[None, :, None, :], key.transpose(-2, -1)
        )
        lag_logits = torch.einsum(
            "bhtd,hld->bhtl", query + self.position_bias[None, :, None, :], relative
        )
        indices = torch.arange(frames, device=x.device)
        lags = (indices[:, None] - indices[None, :]).clamp(0, lag_count - 1)
        position_logits = lag_logits.gather(
            -1, lags[None, None, :, :].expand(batch, heads, -1, -1)
        )
        scores = (content_logits + position_logits) / math.sqrt(head_dim)
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
        # Half precision softmax can otherwise overflow on long recordings.
        softmax_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
        weights = torch.softmax(scores, dim=-1, dtype=softmax_dtype).to(dtype=value.dtype)
        weights = functional.dropout(weights, self.attention.dropout, self.training)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, frames, hidden)
        return self.attention.out_proj(attended)

    def forward(self, x: Tensor) -> Tensor:
        # Input/output: [B*F_sub, T, H]
        x = x + 0.5 * self.ffn1(x)
        normalized = self.attention_norm(x)
        mask = self._attention_mask(x.shape[1], x.device)
        if self.relative_position:
            attended = self._relative_attention(normalized, mask)
        else:
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=mask,
                need_weights=False,
            )
        x = x + self.attention_dropout(attended)
        x = x + self.convolution(x)
        x = x + 0.5 * self.ffn2(x)
        return self.output_norm(x)


class SubBandConformer(nn.Module):
    """Paper Fig. 1(d): shared temporal Conformer over downsampled sub-bands."""

    def __init__(self, config: CabinSepConfig) -> None:
        super().__init__()
        channels = config.fusion_channels
        hidden = config.subband_hidden
        kernel = config.subband_frequency_kernel
        stride = config.subband_frequency_stride
        frequency_bins = config.num_frequency_bins
        padded_bins = _right_padded_size(frequency_bins, kernel, stride)

        self.channels = channels
        self.hidden = hidden
        self.frequency_bins = frequency_bins
        self.padded_bins = padded_bins
        self.downsample = nn.Conv2d(
            channels,
            hidden,
            kernel_size=(1, kernel),
            stride=(1, stride),
        )
        # [REFERENCE] PReLU+cGLN follow the sub-band block in reference [30].
        self.prelu = nn.PReLU(hidden)
        self.norm = CausalGlobalLayerNormMap(hidden)
        self.conformer = nn.ModuleList(
            [
                CausalConformerBlock(
                    hidden=hidden,
                    ffn_dim=config.conformer_ffn_dim,
                    heads=config.conformer_heads,
                    conv_kernel=config.conformer_conv_kernel,
                    dropout=config.conformer_dropout,
                    left_context_frames=config.conformer_left_context_frames,
                    relative_position=config.conformer_relative_position,
                    convolution_norm=config.conformer_conv_norm,
                )
                for _ in range(config.conformer_layers)
            ]
        )
        self.upsample = nn.ConvTranspose2d(
            hidden,
            channels,
            kernel_size=(1, kernel),
            stride=(1, stride),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input/output: [B, C, T, F]
        _assert_feature_map(x, self.channels)
        if x.shape[-1] != self.frequency_bins:
            raise ValueError(
                f"SubBandConformer was built for F={self.frequency_bins}, got F={x.shape[-1]}"
            )
        residual = x
        x = functional.pad(x, (0, self.padded_bins - self.frequency_bins, 0, 0))
        x = self.norm(self.prelu(self.downsample(x)))
        # [B, H, T, F_sub] -> [B*F_sub, T, H]
        sequences, shape = to_subband_sequences(x)
        for layer in self.conformer:
            sequences = layer(sequences)
        # [B*F_sub, T, H] -> [B, H, T, F_sub]
        x = from_subband_sequences(sequences, shape)
        x = self.upsample(x)[..., : self.frequency_bins]
        return residual + x


class FullSubModule(nn.Module):
    """Paper Sec. 3.3: full-band LSTM -> time-skip TAC -> sub-band Conformer."""

    def __init__(self, config: CabinSepConfig) -> None:
        super().__init__()
        self.channels = config.fusion_channels
        self.time_skip = config.time_skip
        self.inference_offset = config.inference_time_skip_offset
        self.full_band = FullBandLSTM(config)
        self.tac = TACBlock(
            channels=config.fusion_channels,
            frequency_bins=config.num_frequency_bins,
            compression_ratio=config.tac_compression_ratio,
            axis=config.tac_axis,
        )
        self.sub_band = SubBandConformer(config)
        self.last_time_skip_offset = 0

    def _time_skip_tac(self, x: Tensor) -> Tensor:
        # Input/output: [B, C, T, F]. Only offset::2 frames are transformed.
        if not self.time_skip:
            return self.tac(x)
        frames = x.shape[2]
        if self.training and frames > 1:
            offset = int(torch.randint(0, 2, (), device=x.device).item())
        else:
            offset = self.inference_offset
        if offset not in (0, 1):
            raise ValueError("Time-skip offset must be 0 or 1")
        self.last_time_skip_offset = offset
        if offset >= frames:
            # An odd-parity, single-frame prefix has no selected frames.  Do
            # not switch parity: doing so changes the prefix when it grows.
            return x

        indices = torch.arange(offset, frames, 2, device=x.device)
        selected = x.index_select(2, indices)
        processed = self.tac(selected)
        # Differentiable replacement: skipped parity remains exactly unchanged.
        return x.index_copy(2, indices, processed)

    def forward(self, x: Tensor) -> Tensor:
        _assert_feature_map(x, self.channels)
        x = self.full_band(x)
        x = self._time_skip_tac(x)
        return self.sub_band(x)


# ============================================================
# Mask estimation (Paper Fig. 1(e))
# ============================================================


class MaskEstimator(nn.Module):
    """Dual speech/noise mask head with frequency-axis linear layers."""

    def __init__(self, config: CabinSepConfig) -> None:
        super().__init__()
        bins = config.num_frequency_bins
        noise_hidden = config.mask_noise_hidden or bins
        self.zones = config.num_zones
        self.frequency_bins = bins

        # Speech: LayerNorm -> Linear -> Sigmoid.
        self.speech_norm = nn.LayerNorm(bins)
        self.speech_linear = nn.Linear(bins, bins)

        # Noise: LayerNorm -> Linear -> GLU -> Linear -> ReLU.
        self.noise_norm = nn.LayerNorm(bins)
        self.noise_linear_in = nn.Linear(bins, 2 * noise_hidden)
        self.noise_linear_out = nn.Linear(noise_hidden, bins)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Input/output masks: [B, Z, T, F]. LayerNorm/Linear act on F.
        _assert_feature_map(x, self.zones)
        if x.shape[-1] != self.frequency_bins:
            raise ValueError(f"MaskEstimator expected F={self.frequency_bins}, got {x.shape[-1]}")
        speech_mask = torch.sigmoid(self.speech_linear(self.speech_norm(x)))
        noise_hidden = functional.glu(self.noise_linear_in(self.noise_norm(x)), dim=-1)
        noise_mask = functional.relu(self.noise_linear_out(noise_hidden))
        return speech_mask, noise_mask


# ============================================================
# Inference-only streaming mask-based MVDR
# ============================================================


@dataclass
class MVDRState:
    """Causal covariance accumulators, reusable across consecutive chunks.

    Numerators: [B, S, F, M, M]; weights: [B, S, F].  S is the number of
    target zones and M is the number of microphones (both are Z in CabinSep).
    """

    speech_numerator: Tensor
    noise_numerator: Tensor
    speech_weight: Tensor
    noise_weight: Tensor

    def detach(self) -> "MVDRState":
        return MVDRState(
            self.speech_numerator.detach(),
            self.noise_numerator.detach(),
            self.speech_weight.detach(),
            self.noise_weight.detach(),
        )


class StreamingMVDR(nn.Module):
    """Causal mask-based reference-channel MVDR used only at inference.

    Paper Eq. (4) is implemented as the standard Souden form
    ``noise_cov^-1 @ speech_cov @ e / trace(noise_cov^-1 @ speech_cov)``.
    The paper's sentence naming Psi/Phi appears reversed relative to this
    equation.  A loaded Hermitian solve is used, with a reference-channel
    fallback for any frequency whose Cholesky factorization fails.

    [ASSUMPTION] CabinSep delegates the exact frame update to reference [31],
    whose Woodbury update is not specified in CabinSep.  This implementation
    maintains cumulative/exponentially-forgotten causal covariance statistics;
    it is a transparent streaming approximation, not a claim of bit-exact [31].
    """

    def __init__(
        self,
        num_microphones: int,
        forgetting_factor: float = 1.0,
        diagonal_loading: float = 1.0e-4,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if num_microphones < 1:
            raise ValueError("num_microphones must be positive")
        if not 0.0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if not math.isfinite(diagonal_loading) or diagonal_loading < 0.0:
            raise ValueError("diagonal_loading must be finite and nonnegative")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self.num_microphones = num_microphones
        self.forgetting_factor = forgetting_factor
        self.diagonal_loading = diagonal_loading
        self.epsilon = epsilon

    def _initial_state(self, spectrum: Tensor, sources: int) -> MVDRState:
        batch, microphones, _, bins = spectrum.shape
        matrix_shape = (batch, sources, bins, microphones, microphones)
        weight_shape = (batch, sources, bins)
        return MVDRState(
            speech_numerator=torch.zeros(matrix_shape, dtype=spectrum.dtype, device=spectrum.device),
            noise_numerator=torch.zeros(matrix_shape, dtype=spectrum.dtype, device=spectrum.device),
            speech_weight=torch.zeros(weight_shape, dtype=spectrum.real.dtype, device=spectrum.device),
            noise_weight=torch.zeros(weight_shape, dtype=spectrum.real.dtype, device=spectrum.device),
        )

    def _validate_state(
        self, state: MVDRState, batch: int, sources: int, bins: int, microphones: int
    ) -> None:
        matrix_shape = (batch, sources, bins, microphones, microphones)
        weight_shape = (batch, sources, bins)
        if tuple(state.speech_numerator.shape) != matrix_shape:
            raise ValueError("MVDR speech state is incompatible with the current input")
        if tuple(state.noise_numerator.shape) != matrix_shape:
            raise ValueError("MVDR noise state is incompatible with the current input")
        if tuple(state.speech_weight.shape) != weight_shape:
            raise ValueError("MVDR speech-weight state is incompatible with the current input")
        if tuple(state.noise_weight.shape) != weight_shape:
            raise ValueError("MVDR noise-weight state is incompatible with the current input")

    def forward(
        self,
        spectrum: Tensor,
        speech_mask: Tensor,
        noise_mask: Tensor,
        state: Optional[MVDRState] = None,
        reference_microphones: Optional[Sequence[int]] = None,
    ) -> Tuple[Tensor, MVDRState]:
        # spectrum: [B, M, T, F]; masks: [B, S, T, F]
        if spectrum.ndim != 4 or not spectrum.is_complex():
            raise ValueError("spectrum must be complex [B, M, T, F]")
        if speech_mask.shape != noise_mask.shape or speech_mask.ndim != 4:
            raise ValueError("speech/noise masks must have equal [B, S, T, F] shapes")
        batch, microphones, frames, bins = spectrum.shape
        mask_batch, sources, mask_frames, mask_bins = speech_mask.shape
        if (mask_batch, mask_frames, mask_bins) != (batch, frames, bins):
            raise ValueError("Mask time-frequency dimensions do not match spectrum")
        if microphones != self.num_microphones:
            raise ValueError(
                f"StreamingMVDR expects {self.num_microphones} microphones, got {microphones}"
            )
        if min(batch, sources, frames, bins) <= 0:
            raise ValueError("MVDR batch, source, frame and frequency dimensions must be nonempty")
        if not speech_mask.is_floating_point() or not noise_mask.is_floating_point():
            raise TypeError("MVDR masks must be real floating-point tensors")
        if speech_mask.device != spectrum.device or noise_mask.device != spectrum.device:
            raise ValueError("MVDR masks and spectrum must be on the same device")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in (spectrum, speech_mask, noise_mask)):
            raise ValueError("MVDR input contains NaN or infinity")
        speech_mask = speech_mask.to(dtype=spectrum.real.dtype)
        noise_mask = noise_mask.to(dtype=spectrum.real.dtype)

        if reference_microphones is None:
            if sources > microphones:
                raise ValueError("A default reference microphone is unavailable for every source")
            reference_microphones = list(range(sources))
        if len(reference_microphones) != sources:
            raise ValueError("reference_microphones must contain one index per source")
        if min(reference_microphones) < 0 or max(reference_microphones) >= microphones:
            raise ValueError("reference_microphones contains an invalid microphone index")

        if state is None:
            state = self._initial_state(spectrum, sources)
        self._validate_state(state, batch, sources, bins, microphones)
        for name in ("speech_numerator", "noise_numerator", "speech_weight", "noise_weight"):
            value = getattr(state, name)
            expected_dtype = spectrum.dtype if name.endswith("numerator") else spectrum.real.dtype
            if value.device != spectrum.device or value.dtype != expected_dtype:
                raise ValueError(f"MVDR {name} state dtype/device does not match the spectrum")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"MVDR {name} state contains NaN or infinity")

        references = functional.one_hot(
            torch.as_tensor(reference_microphones, device=spectrum.device),
            num_classes=microphones,
        ).to(dtype=spectrum.dtype)
        identity = torch.eye(microphones, dtype=spectrum.dtype, device=spectrum.device)
        output_frames: List[Tensor] = []

        speech_numerator = state.speech_numerator
        noise_numerator = state.noise_numerator
        speech_weight = state.speech_weight
        noise_weight = state.noise_weight

        for frame in range(frames):
            # Microphone vector y: [B, F, M]. Outer product yy^H: [B, F, M, M].
            y = spectrum[:, :, frame, :].permute(0, 2, 1)
            outer = y.unsqueeze(-1) * y.conj().unsqueeze(-2)
            speech_frame_mask = speech_mask[:, :, frame, :].clamp_min(0.0)
            noise_frame_mask = noise_mask[:, :, frame, :].clamp_min(0.0)

            factor = self.forgetting_factor
            speech_numerator = (
                factor * speech_numerator
                + speech_frame_mask[..., None, None] * outer[:, None, ...]
            )
            noise_numerator = (
                factor * noise_numerator
                + noise_frame_mask[..., None, None] * outer[:, None, ...]
            )
            speech_weight = factor * speech_weight + speech_frame_mask
            noise_weight = factor * noise_weight + noise_frame_mask

            speech_covariance = speech_numerator / (
                speech_weight[..., None, None] + self.epsilon
            )
            noise_covariance = noise_numerator / (
                noise_weight[..., None, None] + self.epsilon
            )

            # An absolute epsilon alone can round away for energetic,
            # rank-deficient channels. Enforce a dtype-relative loading floor
            # even if the configurable regularization coefficient is zero.
            noise_trace = noise_covariance.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
            relative_loading = max(self.diagonal_loading, 10.0 * torch.finfo(spectrum.real.dtype).eps)
            loading = self.epsilon + relative_loading * noise_trace.clamp_min(0.0) / microphones
            loaded_noise = noise_covariance + loading[..., None, None] * identity

            # A = Psi^-1 Phi, with Psi=noise and Phi=target speech.
            # cholesky_ex reports failures per matrix, so one unstable bin does
            # not terminate an entire recording. No inverse is materialized.
            cholesky, info = torch.linalg.cholesky_ex(loaded_noise, check_errors=False)
            solved = (info == 0) & torch.isfinite(cholesky).flatten(-2).all(dim=-1)
            safe_cholesky = torch.where(solved[..., None, None], cholesky, identity)
            transfer = torch.cholesky_solve(speech_covariance, safe_cholesky)
            numerator = torch.einsum("bsfmn,sn->bsfm", transfer, references)
            denominator = transfer.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            valid = (
                solved
                & (denominator.abs() > self.epsilon)
                & (speech_weight > self.epsilon)
                & torch.isfinite(denominator)
                & torch.isfinite(numerator).all(dim=-1)
            )
            safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
            beamformer = numerator / safe_denominator.unsqueeze(-1)
            fallback = references[None, :, None, :].expand(batch, -1, bins, -1)
            beamformer = torch.where(valid.unsqueeze(-1), beamformer, fallback)

            # Paper Eq. (3): X_i,t,f = W_i,t,f^H Y_t,f -> [B, S, F].
            output_frames.append(torch.einsum("bsfm,bfm->bsf", beamformer.conj(), y))

        enhanced = torch.stack(output_frames, dim=2)  # [B, S, T, F]
        new_state = MVDRState(
            speech_numerator,
            noise_numerator,
            speech_weight,
            noise_weight,
        )
        return enhanced, new_state


# ============================================================
# CabinSep orchestration
# ============================================================


class CabinSep(nn.Module):
    """CabinSep mask estimator with distinct training and MVDR inference paths."""

    def __init__(self, config: CabinSepConfig) -> None:
        super().__init__()
        self.config = config.validate()
        self.spec_encoder = SpecEncoder(config)
        self.lps_encoder = LPSEncoder(config)
        self.ipd_encoder = IPDEncoder(config)

        concatenated_channels = 3 * config.encoder_output_channels
        fusion_padding = (config.fusion_kernel[0] // 2, config.fusion_kernel[1] // 2)
        self.fusion = nn.Conv2d(
            concatenated_channels,
            config.fusion_channels,
            kernel_size=config.fusion_kernel,
            padding=fusion_padding,
        )
        self.full_sub_modules = nn.ModuleList(
            [FullSubModule(config) for _ in range(config.num_full_sub_modules)]
        )
        # [ASSUMPTION]/[REFERENCE] Kernel 1x3 mirrors the pre-Full-Sub Conv2d.
        self.mask_feature_decoder = nn.ConvTranspose2d(
            config.fusion_channels,
            config.num_zones,
            kernel_size=config.fusion_kernel,
            padding=fusion_padding,
        )
        self.mask_estimator = MaskEstimator(config)
        self.mvdr = StreamingMVDR(
            num_microphones=config.num_zones,
            forgetting_factor=config.mvdr_forgetting_factor,
            diagonal_loading=config.mvdr_diagonal_loading,
            epsilon=config.mvdr_epsilon,
        )
        self.register_buffer(
            "analysis_window",
            torch.hamming_window(config.window_length, periodic=True),
            persistent=False,
        )

    def _pad_waveform_for_stft(self, waveform: Tensor) -> Tensor:
        length = waveform.shape[-1]
        target = max(length, self.config.n_fft)
        remainder = (target - self.config.n_fft) % self.config.hop_length
        target += (self.config.hop_length - remainder) % self.config.hop_length
        return functional.pad(waveform, (0, target - length))

    def stft(self, waveform: Tensor) -> Tensor:
        """Waveform [B, Z, L] -> complex spectrum [B, Z, T, F]."""

        if waveform.ndim != 3:
            raise ValueError(f"Expected mixture [B, Z, L], got {tuple(waveform.shape)}")
        if waveform.shape[1] != self.config.num_zones:
            raise ValueError(
                f"Expected Z={self.config.num_zones}, got Z={waveform.shape[1]}"
            )
        if not waveform.is_floating_point():
            raise TypeError("Waveform must be floating point")
        if waveform.shape[0] == 0 or waveform.shape[-1] == 0:
            raise ValueError("Waveform batch and sample dimensions must be nonempty")
        # FFT backends cannot consistently handle half/bfloat16 waveforms.
        if waveform.dtype in (torch.float16, torch.bfloat16):
            waveform = waveform.float()
        waveform = self._pad_waveform_for_stft(waveform)
        batch, zones, samples = waveform.shape
        flat = waveform.reshape(batch * zones, samples)
        window = self.analysis_window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.window_length,
            window=window,
            center=self.config.stft_center,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        # torch.stft [B*Z, F, T] -> canonical [B, Z, T, F].
        return spectrum.reshape(batch, zones, spectrum.shape[-2], spectrum.shape[-1]).permute(
            0, 1, 3, 2
        ).contiguous()

    def istft(self, spectrum: Tensor, length: int) -> Tensor:
        """Complex spectrum [B, Z, T, F] -> waveform [B, Z, L]."""

        if spectrum.ndim != 4 or not spectrum.is_complex():
            raise ValueError("Expected complex spectrum [B, Z, T, F]")
        if length <= 0:
            raise ValueError("Requested waveform length must be positive")
        batch, zones, frames, bins = spectrum.shape
        if bins != self.config.num_frequency_bins:
            raise ValueError(f"Expected F={self.config.num_frequency_bins}, got F={bins}")
        flat = spectrum.permute(0, 1, 3, 2).contiguous().reshape(batch * zones, bins, frames)
        window = self.analysis_window.to(device=spectrum.device, dtype=spectrum.real.dtype)
        waveform = torch.istft(
            flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.window_length,
            window=window,
            center=self.config.stft_center,
            normalized=False,
            onesided=True,
            length=length,
            return_complex=False,
        )
        return waveform.reshape(batch, zones, length)

    def extract_features(self, spectrum: Tensor) -> Dict[str, Tensor]:
        """Paper Sec. 3.2 feature extraction from [B, Z, T, F]."""

        if spectrum.ndim != 4 or not spectrum.is_complex():
            raise ValueError("Expected complex spectrum [B, Z, T, F]")
        if spectrum.shape[1] != self.config.num_zones:
            raise ValueError("Spectrum zone count does not match the model")

        # Paper: Y_R in R^(2Z x T x F).
        spec = torch.cat((spectrum.real, spectrum.imag), dim=1)
        # Paper LPS; log(real^2 + imag^2) avoids the paper's Y_R notation typo.
        lps = torch.log(spectrum.abs().square() + self.config.feature_epsilon)
        # Paper Eq. (5): front-row zones 1 and 2 -> Python indices 0 and 1.
        first, second = self.config.front_ipd_pair
        phase_difference = torch.angle(spectrum[:, first]) - torch.angle(spectrum[:, second])
        ipd = torch.stack((torch.cos(phase_difference), torch.sin(phase_difference)), dim=1)
        return {"spec": spec, "lps": lps, "ipd": ipd}

    def estimate_masks_from_spectrum(
        self, spectrum: Tensor, return_intermediates: bool = False
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        features = self.extract_features(spectrum)
        spec_embedding = self.spec_encoder(features["spec"])
        lps_embedding = self.lps_encoder(features["lps"])
        ipd_embedding = self.ipd_encoder(features["ipd"])
        concatenated = torch.cat((spec_embedding, lps_embedding, ipd_embedding), dim=1)
        embedding = self.fusion(concatenated)
        fused_embedding = embedding
        full_sub_outputs: List[Tensor] = []
        for module in self.full_sub_modules:
            embedding = module(embedding)
            if return_intermediates:
                full_sub_outputs.append(embedding)
        decoded = self.mask_feature_decoder(embedding)
        speech_mask, noise_mask = self.mask_estimator(decoded)

        debug: Dict[str, Tensor] = {}
        if return_intermediates:
            debug = {
                **features,
                "spec_embedding": spec_embedding,
                "lps_embedding": lps_embedding,
                "ipd_embedding": ipd_embedding,
                "concatenated_embedding": concatenated,
                "fused_embedding": fused_embedding,
                "final_full_sub_embedding": embedding,
                "decoded_mask_features": decoded,
            }
            for index, output in enumerate(full_sub_outputs):
                debug[f"full_sub_{index + 1}"] = output
        return speech_mask, noise_mask, debug

    def estimate_masks(
        self, mixture: Tensor, return_intermediates: bool = False
    ) -> Dict[str, object]:
        spectrum = self.stft(mixture)
        speech_mask, noise_mask, debug = self.estimate_masks_from_spectrum(
            spectrum, return_intermediates=return_intermediates
        )
        result: Dict[str, object] = {
            "spectrum": spectrum,
            "speech_mask": speech_mask,
            "noise_mask": noise_mask,
        }
        if return_intermediates:
            result["intermediates"] = debug
        return result

    def forward_train(self, mixture: Tensor) -> Dict[str, Tensor]:
        """Differentiable mask-training path; MVDR is intentionally absent."""

        original_length = mixture.shape[-1]
        estimates = self.estimate_masks(mixture)
        spectrum = estimates["spectrum"]
        speech_mask = estimates["speech_mask"]
        noise_mask = estimates["noise_mask"]
        if not isinstance(spectrum, Tensor) or not isinstance(speech_mask, Tensor):
            raise RuntimeError("Unexpected mask-estimation result")
        if not isinstance(noise_mask, Tensor):
            raise RuntimeError("Unexpected mask-estimation result")
        speech_spectrum = speech_mask * spectrum
        noise_spectrum = noise_mask * spectrum
        return {
            "spectrum": spectrum,
            "speech_mask": speech_mask,
            "noise_mask": noise_mask,
            "speech_spectrum": speech_spectrum,
            "noise_spectrum": noise_spectrum,
            "speech": self.istft(speech_spectrum, original_length),
            "noise": self.istft(noise_spectrum, original_length),
        }

    def forward(self, mixture: Tensor) -> Dict[str, Tensor]:
        return self.forward_train(mixture)

    def separate(
        self,
        mixture: Tensor,
        mvdr_state: Optional[MVDRState] = None,
    ) -> Dict[str, object]:
        """Whole-recording inference: causal masks -> recursive MVDR -> iSTFT.

        Call in eval mode under torch.no_grad(). ``mvdr_state`` continues only
        the covariance statistics; neural LSTM/normalization/attention and
        STFT overlap state are not cached. Splitting audio and calling this
        repeatedly is therefore not equivalent to full streaming inference.
        """

        original_length = mixture.shape[-1]
        estimates = self.estimate_masks(mixture)
        spectrum = estimates["spectrum"]
        speech_mask = estimates["speech_mask"]
        noise_mask = estimates["noise_mask"]
        if not all(isinstance(item, Tensor) for item in (spectrum, speech_mask, noise_mask)):
            raise RuntimeError("Unexpected mask-estimation result")
        enhanced_spectrum, new_state = self.mvdr(
            spectrum,  # type: ignore[arg-type]
            speech_mask,  # type: ignore[arg-type]
            noise_mask,  # type: ignore[arg-type]
            state=mvdr_state,
        )
        return {
            **estimates,
            "enhanced_spectrum": enhanced_spectrum,
            "waveform": self.istft(enhanced_spectrum, original_length),
            "mvdr_state": new_state,
        }


# ============================================================
# Paper Eq. (7) training loss
# ============================================================


def _hz_to_mel(frequency: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _make_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    num_mels: int,
    minimum_hz: float,
    maximum_hz: float,
) -> Tensor:
    frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_min = _hz_to_mel(torch.tensor(minimum_hz))
    mel_max = _hz_to_mel(torch.tensor(maximum_hz))
    mel_points = torch.linspace(mel_min, mel_max, num_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    left = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    right = hz_points[2:, None]
    rising = (frequencies[None, :] - left) / (center - left).clamp_min(1.0e-12)
    falling = (right - frequencies[None, :]) / (right - center).clamp_min(1.0e-12)
    return torch.minimum(rising, falling).clamp_min(0.0)


class CabinSepLoss(nn.Module):
    """Paper Eq. (7): speech FBank-MAE + SI-SNR + noise FBank-MAE."""

    def __init__(
        self,
        sample_rate: int,
        config: Optional[LossConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or LossConfig()
        self.sample_rate = sample_rate
        weights = (self.config.speech_fbank_weight, self.config.speech_si_snr_weight,
                   self.config.noise_fbank_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError("Loss weights must be finite, nonnegative, and not all zero")
        if sample_rate <= 0 or self.config.fbank_n_fft <= 0 or self.config.fbank_num_mels <= 0:
            raise ValueError("Sample rate, FBank FFT size, and mel count must be positive")
        if not math.isfinite(self.config.epsilon) or self.config.epsilon <= 0:
            raise ValueError("Loss epsilon must be finite and positive")
        if any(not math.isfinite(value) or value <= 0 for value in (
            self.config.fbank_window_ms, self.config.fbank_hop_ms
        )):
            raise ValueError("FBank window and hop durations must be finite and positive")
        self.fbank_window_length = int(
            round(sample_rate * self.config.fbank_window_ms / 1000.0)
        )
        self.fbank_hop_length = int(round(sample_rate * self.config.fbank_hop_ms / 1000.0))
        if not 0 < self.fbank_hop_length <= self.fbank_window_length <= self.config.fbank_n_fft:
            raise ValueError("FBank requires 0 < hop <= window <= n_fft")
        maximum_hz = (sample_rate / 2.0 if self.config.fbank_max_hz is None
                      else self.config.fbank_max_hz)
        if not 0.0 <= self.config.fbank_min_hz < maximum_hz <= sample_rate / 2.0:
            raise ValueError("Invalid FBank frequency range")
        self.register_buffer(
            "fbank_window",
            torch.hamming_window(self.fbank_window_length, periodic=True),
            persistent=False,
        )
        self.register_buffer(
            "mel_filterbank",
            _make_mel_filterbank(
                sample_rate,
                self.config.fbank_n_fft,
                self.config.fbank_num_mels,
                self.config.fbank_min_hz,
                maximum_hz,
            ),
            persistent=False,
        )

    def _log_fbank(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 3 or min(waveform.shape) <= 0:
            raise ValueError("FBank input must be nonempty [B, Z, L]")
        batch, zones, samples = waveform.shape
        # Frame the actual window before FFT zero-padding. torch.stft with
        # win_length < n_fft, center=False would discard the first samples and
        # truncate the final partial frame, leaving short clips unsupervised.
        padded_samples = _right_padded_size(
            samples, self.fbank_window_length, self.fbank_hop_length
        )
        waveform = functional.pad(waveform, (0, padded_samples - samples))
        frames = waveform.reshape(batch * zones, padded_samples).unfold(
            -1, self.fbank_window_length, self.fbank_hop_length
        )
        window = self.fbank_window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.fft.rfft(frames * window, n=self.config.fbank_n_fft, dim=-1)
        power = spectrum.abs().square()  # [B*Z, T, F]
        mel = torch.matmul(
            power,
            self.mel_filterbank.to(device=waveform.device, dtype=waveform.dtype).transpose(0, 1),
        )
        return torch.log(mel + self.config.epsilon).reshape(
            batch, zones, mel.shape[1], mel.shape[2]
        )

    def fbank_mae(self, estimate: Tensor, target: Tensor) -> Tensor:
        return functional.l1_loss(self._log_fbank(estimate), self._log_fbank(target))

    def si_snr_loss(self, estimate: Tensor, target: Tensor) -> Tensor:
        """Negative SI-SNR: active-zone mean per example, then batch mean.

        Inactive targets have no defined SI-SNR and contribute zero here;
        both FBank terms still supervise those zones.
        """

        if estimate.shape != target.shape or estimate.ndim != 3:
            raise ValueError("SI-SNR inputs must have equal [B, Z, L] shapes")
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        target_energy = target.square().sum(dim=-1, keepdim=True)
        projection = (
            (estimate * target).sum(dim=-1, keepdim=True)
            / (target_energy + self.config.epsilon)
        ) * target
        residual = estimate - projection
        ratio = (projection.square().sum(dim=-1) + self.config.epsilon) / (
            residual.square().sum(dim=-1) + self.config.epsilon
        )
        si_snr = 10.0 * torch.log10(ratio + self.config.epsilon)
        active = target_energy.squeeze(-1) > self.config.epsilon
        per_example = (si_snr * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1)
        return -per_example.mean()

    def forward(
        self,
        estimated_speech: Tensor,
        target_speech: Tensor,
        estimated_noise: Tensor,
        target_noise: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if estimated_speech.shape != target_speech.shape:
            raise ValueError("Estimated and target speech shapes differ")
        if estimated_noise.shape != target_noise.shape:
            raise ValueError("Estimated and target noise shapes differ")
        if estimated_speech.shape != estimated_noise.shape:
            raise ValueError("Speech and noise must share the same [B, Z, L] shape")
        if estimated_speech.ndim != 3 or min(estimated_speech.shape) <= 0:
            raise ValueError("Loss inputs must be nonempty [B, Z, L] tensors")
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=estimated_speech.device)
            if lengths.ndim != 1 or lengths.numel() != estimated_speech.shape[0]:
                raise ValueError("lengths must have shape [B]")
            if lengths.dtype == torch.bool or lengths.is_complex() or (
                lengths.is_floating_point() and not bool((lengths == lengths.round()).all())
            ):
                raise ValueError("lengths must contain integer sample counts")
            if not bool(((lengths > 0) & (lengths <= estimated_speech.shape[-1])).all()):
                raise ValueError("lengths must be between 1 and the padded waveform length")
            # Group equal lengths: crop before FBank framing and SI-SNR means.
            # Padding cannot contribute loss or gradients, even at frame edges.
            totals: Dict[str, Tensor] = {}
            for valid_samples in lengths.unique().tolist():
                selected = lengths == valid_samples
                count = int(selected.sum().item())
                stop = int(valid_samples)
                values = self.forward(
                    estimated_speech[selected, :, :stop], target_speech[selected, :, :stop],
                    estimated_noise[selected, :, :stop], target_noise[selected, :, :stop],
                )
                for name, value in values.items():
                    weighted = value * (count / estimated_speech.shape[0])
                    totals[name] = totals.get(name, value.new_zeros(())) + weighted
            return totals
        speech_fbank = self.fbank_mae(estimated_speech, target_speech)
        speech_si_snr = self.si_snr_loss(estimated_speech, target_speech)
        noise_fbank = self.fbank_mae(estimated_noise, target_noise)
        total = (
            self.config.speech_fbank_weight * speech_fbank
            + self.config.speech_si_snr_weight * speech_si_snr
            + self.config.noise_fbank_weight * noise_fbank
        )
        return {
            "loss": total,
            "speech_fbank_mae": speech_fbank,
            "speech_si_snr_loss": speech_si_snr,
            "noise_fbank_mae": noise_fbank,
        }


if __name__ == "__main__":
    # Opening model.py and pressing Run performs a dataset-free model check.
    from smoke_test import main

    main()
