"""Run-button CabinSep inference.

Edit ``INFERENCE_RUN`` in config.py, open this file, and press Run.  The input
must be one multi-channel audio file whose channel order is zone 1..Z.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from config import CabinSepConfig, INFERENCE_RUN, InferenceRunConfig
from model import CabinSep


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("CabinSep complex STFT/MVDR requires a CPU or CUDA device")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}\n"
            "Edit INFERENCE_RUN.checkpoint_path in config.py."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Not a CabinSep checkpoint: {path}")
    return checkpoint


def _model_config(
    checkpoint: Mapping[str, Any], run_config: InferenceRunConfig
) -> CabinSepConfig:
    serialized = checkpoint.get("model_config")
    if serialized is None:
        print(
            "WARNING: checkpoint has no model_config; using "
            f"CabinSep-{run_config.fallback_model_variant} from config.py"
        )
        return CabinSepConfig.from_variant(run_config.fallback_model_variant)
    if not isinstance(serialized, Mapping):
        raise ValueError("checkpoint['model_config'] must be a mapping")
    values = dict(serialized)
    # Preserve the architecture of checkpoints produced before the paper audit.
    # New training always serializes all three explicit options.
    legacy_defaults = {"tac_axis": "frequency", "time_skip": True,
                       "conformer_relative_position": False}
    for name, default in legacy_defaults.items():
        if name not in values:
            values[name] = default
            print(f"Legacy checkpoint: using {name}={default!r}")
    if "conformer_conv_norm" not in values:
        state = checkpoint.get("model_state_dict", {})
        values["conformer_conv_norm"] = (
            "batch" if any(".convolution.batch_norm." in key for key in state) else "layer"
        )
        print(f"Legacy checkpoint: using conformer_conv_norm={values['conformer_conv_norm']!r}")
    return CabinSepConfig(**values).validate()


def run_inference(run_config: InferenceRunConfig = INFERENCE_RUN) -> Path:
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("Inference audio I/O requires `pip install soundfile`") from error

    checkpoint_path = _project_path(run_config.checkpoint_path)
    input_path = _project_path(run_config.input_path)
    output_directory = _project_path(run_config.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input audio not found: {input_path.resolve()}\n"
            "Edit INFERENCE_RUN.input_path in config.py."
        )

    checkpoint = _load_checkpoint(checkpoint_path)
    model_config = _model_config(checkpoint, run_config)
    device = resolve_device(run_config.device)

    # soundfile returns [L, channels]. Canonical CabinSep layout is [B, Z, L].
    audio, sample_rate = sf.read(input_path, dtype="float32", always_2d=True)
    if sample_rate != model_config.sample_rate:
        raise ValueError(
            f"Input sample rate is {sample_rate} Hz, but checkpoint expects "
            f"{model_config.sample_rate} Hz. Resample in the data/audio preparation "
            "step; inference intentionally does not hide resampling."
        )
    if audio.shape[1] != model_config.num_zones:
        raise ValueError(
            f"Input has {audio.shape[1]} channels, but checkpoint expects "
            f"{model_config.num_zones}. Channel order must be zone 1..Z."
        )
    if audio.shape[0] == 0:
        raise ValueError("Input audio is empty")
    for zone_index in range(model_config.num_zones):
        if (output_directory / f"zone_{zone_index + 1}.wav").resolve() == input_path.resolve():
            raise ValueError("Output zone WAV would overwrite the input; choose another output_dir")

    mixture = torch.from_numpy(audio.T.copy()).unsqueeze(0).to(device)
    if not bool(torch.isfinite(mixture).all()):
        raise ValueError("Input audio contains NaN or infinity")
    model = CabinSep(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"Device: {device}")
    print(f"Model: CabinSep-{model_config.variant}")
    print(f"Input: {input_path.resolve()} ({audio.shape[0]} samples, {audio.shape[1]} zones)")
    with torch.inference_mode():
        result = model.separate(mixture)
    waveform = result["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise RuntimeError("CabinSep returned an invalid waveform")
    waveform = waveform.squeeze(0).detach().cpu()
    if not bool(torch.isfinite(waveform).all()):
        raise FloatingPointError("Inference output contains NaN or infinity")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_files = []
    for zone_index in range(model_config.num_zones):
        output_path = output_directory / f"zone_{zone_index + 1}.wav"
        # FLOAT avoids clipping MVDR samples whose magnitude temporarily exceeds 1.
        sf.write(
            output_path,
            waveform[zone_index].numpy(),
            model_config.sample_rate,
            subtype="FLOAT",
        )
        output_files.append(str(output_path.resolve()))
        print(f"Saved zone {zone_index + 1}: {output_path.resolve()}")

    if run_config.save_masks:
        speech_mask = result["speech_mask"]
        noise_mask = result["noise_mask"]
        if not isinstance(speech_mask, torch.Tensor) or not isinstance(noise_mask, torch.Tensor):
            raise RuntimeError("CabinSep returned invalid masks")
        mask_path = output_directory / "masks.pt"
        torch.save(
            {
                "speech_mask": speech_mask.detach().cpu(),
                "noise_mask": noise_mask.detach().cpu(),
                "layout": "[B, Z, T, F]",
            },
            mask_path,
        )
    else:
        mask_path = None

    manifest = {
        "checkpoint": str(checkpoint_path.resolve()),
        "input": str(input_path.resolve()),
        "sample_rate": model_config.sample_rate,
        "num_zones": model_config.num_zones,
        "num_samples": int(waveform.shape[-1]),
        "output_files": output_files,
        "mask_file": str(mask_path.resolve()) if mask_path is not None else None,
        "model_config": asdict(model_config),
        "processing": "whole-file causal model; waveform chunk state is not cached",
        "mvdr": (
            "Causal cumulative-mask covariance implementation; CabinSep reference "
            "[31] supplies the reference method, but CabinSep omits its exact "
            "dual-mask update settings. This code uses a direct linear solve."
        ),
    }
    manifest_path = output_directory / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Inference complete: {output_directory.resolve()}")
    return output_directory


def main() -> None:
    run_inference(INFERENCE_RUN)


if __name__ == "__main__":
    main()
