"""Run-button training entry point for CabinSep.

Data construction is intentionally outside this repository.  The external
team only needs to provide the factory configured by ``config.TRAIN_RUN``:

    def build_dataloaders(train_config, model_config):
        return {"train": train_loader, "val": optional_validation_loader}

Each loader yields a mapping containing float waveforms ``mixture``, ``speech``
and ``noise`` with shape [B, Z, L] by default.  Edit TRAIN_RUN in config.py and
press Run; no terminal arguments are required.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import numpy as np
from torch import Tensor
from torch.nn import functional

from config import (
    CabinSepConfig,
    LossConfig,
    TRAIN_LOSS,
    TRAIN_MODEL,
    TRAIN_RUN,
    TrainingRunConfig,
)
from model import CabinSep, CabinSepLoss


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("CabinSep's complex STFT/MVDR path supports device='cpu' or 'cuda'")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_factory(specification: str):
    try:
        module_name, function_name = specification.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError(
            "loader_factory must use 'python.module:function_name' syntax"
        ) from error
    if not module_name or not function_name:
        raise ValueError("loader_factory must use 'python.module:function_name' syntax")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Could not import DataLoader module {module_name!r}. The dataset team "
            "should add it without changing the model code, or update "
            "TRAIN_RUN.loader_factory in config.py."
        ) from error
    try:
        factory = getattr(module, function_name)
    except AttributeError as error:
        raise RuntimeError(
            f"DataLoader module {module_name!r} has no function {function_name!r}"
        ) from error
    if not callable(factory):
        raise TypeError(f"DataLoader factory {specification!r} is not callable")
    return factory


def build_external_loaders(
    run_config: TrainingRunConfig,
    model_config: CabinSepConfig,
) -> Tuple[Iterable[Mapping[str, Any]], Optional[Iterable[Mapping[str, Any]]]]:
    """Call the single integration point owned by the dataset team."""

    factory = _load_factory(run_config.loader_factory)
    loaders = factory(run_config, model_config)
    if isinstance(loaders, Mapping):
        if "train" not in loaders:
            raise ValueError("DataLoader factory mapping must contain a 'train' loader")
        return loaders["train"], loaders.get("val")
    if isinstance(loaders, tuple):
        if len(loaders) == 1:
            return loaders[0], None
        if len(loaders) == 2:
            return loaders[0], loaders[1]
        raise ValueError("DataLoader factory tuple must be (train,) or (train, val)")
    return loaders, None


def adapt_batch(
    batch: Mapping[str, Any],
    run_config: TrainingRunConfig,
    model_config: CabinSepConfig,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Return BZL tensors and optional valid lengths; zero padded suffixes."""

    if not isinstance(batch, Mapping):
        raise TypeError("Each training batch must be a mapping with mixture/speech/noise keys")

    required = (run_config.mixture_key, run_config.speech_key, run_config.noise_key)
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Training batch is missing keys: {missing}")

    tensors = []
    for key in required:
        value = batch[key]
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.ndim != 3:
            raise ValueError(f"batch[{key!r}] must be 3-D, got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise TypeError(
                f"batch[{key!r}] must already be normalized floating-point audio; "
                f"got {tensor.dtype}"
            )
        if run_config.batch_layout.upper() == "BLZ":
            tensor = tensor.permute(0, 2, 1)
        elif run_config.batch_layout.upper() != "BZL":
            raise ValueError("batch_layout must be BZL or BLZ")
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"batch[{key!r}] contains NaN or infinity")
        tensors.append(tensor.contiguous())

    mixture, speech, noise = tensors
    if any(size == 0 for size in mixture.shape):
        raise ValueError("Training batch dimensions B, Z, and L must be positive")
    if mixture.shape != speech.shape or mixture.shape != noise.shape:
        raise ValueError(
            "mixture, speech, and noise must have identical [B, Z, L] shapes; "
            f"got {tuple(mixture.shape)}, {tuple(speech.shape)}, {tuple(noise.shape)}"
        )
    if mixture.shape[1] != model_config.num_zones:
        raise ValueError(
            f"Batch has Z={mixture.shape[1]}, model expects Z={model_config.num_zones}"
        )

    if "sample_rate" in batch:
        sample_rates = torch.as_tensor(batch["sample_rate"]).reshape(-1)
        if sample_rates.numel() not in (1, mixture.shape[0]):
            raise ValueError("batch['sample_rate'] must be a scalar or one value per example")
        if not bool((sample_rates == model_config.sample_rate).all()):
            raise ValueError(
                f"Batch sample_rate must be {model_config.sample_rate}; got "
                f"{sample_rates.unique().tolist()}"
            )
    lengths = None
    if run_config.lengths_key is not None and run_config.lengths_key in batch:
        lengths = torch.as_tensor(batch[run_config.lengths_key], device=device)
        if lengths.ndim != 1 or lengths.shape[0] != mixture.shape[0]:
            raise ValueError("lengths must have shape [B], one valid sample count per example")
        if lengths.dtype == torch.bool or lengths.is_complex():
            raise TypeError("lengths must contain integer sample counts")
        if not bool(torch.isfinite(lengths).all()) or not bool((lengths == lengths.long()).all()):
            raise ValueError("lengths must contain finite integer sample counts")
        lengths = lengths.long()
        if bool(((lengths <= 0) | (lengths > mixture.shape[-1])).any()):
            raise ValueError("Each length must be in [1, padded waveform length]")
        valid = torch.arange(mixture.shape[-1], device=device)[None, None, :] < lengths[:, None, None]
        mixture, speech, noise = [tensor.masked_fill(~valid, 0.0) for tensor in tensors]
    return mixture, speech, noise, lengths


def forward_train_batch(
    model: CabinSep, mixture: Tensor, lengths: Optional[Tensor]
) -> Dict[str, Tensor]:
    """Run the model without letting padded tails alter valid boundary samples.

    A center=False overlap-add transform can synthesize valid end samples from
    an extra all-zero frame when a short example is embedded in a longer padded
    batch. Grouping equal lengths and cropping before STFT makes a padded batch
    exactly equivalent to processing its examples at their actual lengths.
    """

    if lengths is None or bool((lengths == mixture.shape[-1]).all()):
        return model.forward_train(mixture)
    if lengths.ndim != 1 or lengths.shape[0] != mixture.shape[0]:
        raise ValueError("lengths must have shape [B]")

    padded_length = mixture.shape[-1]
    separated_speech = mixture.new_zeros(mixture.shape)
    separated_noise = mixture.new_zeros(mixture.shape)
    for valid_samples in lengths.unique(sorted=True).tolist():
        stop = int(valid_samples)
        indices = torch.nonzero(lengths == stop, as_tuple=False).flatten()
        group = mixture.index_select(0, indices)[..., :stop]
        output = model.forward_train(group)
        speech = functional.pad(output["speech"], (0, padded_length - stop))
        noise = functional.pad(output["noise"], (0, padded_length - stop))
        separated_speech = separated_speech.index_copy(0, indices, speech)
        separated_noise = separated_noise.index_copy(0, indices, noise)
    return {"speech": separated_speech, "noise": separated_noise}


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path.resolve()}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions before weights_only.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Not a CabinSep training checkpoint: {path}")
    return checkpoint


def _configuration_from_checkpoint(checkpoint: Mapping[str, Any]) -> CabinSepConfig:
    values = checkpoint.get("model_config")
    if not isinstance(values, Mapping):
        raise ValueError("Checkpoint has no serialized model_config")
    values = dict(values)
    if "tac_axis" not in values:
        values["tac_axis"] = "frequency"
        print("Legacy checkpoint: preserving its frequency-axis TAC interpretation")
    values.setdefault("time_skip", True)
    values.setdefault("conformer_relative_position", False)
    if "conformer_conv_norm" not in values:
        state = checkpoint.get("model_state_dict", {})
        values["conformer_conv_norm"] = (
            "batch" if any(".convolution.batch_norm." in key for key in state) else "layer"
        )
        print(f"Legacy checkpoint: using conformer_conv_norm={values['conformer_conv_norm']!r}")
    return CabinSepConfig(**values).validate()


def _capture_rng_state(loader: Any = None) -> Dict[str, Any]:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }
    for name, owner in (("loader_generator", loader), ("sampler_generator", getattr(loader, "sampler", None))):
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator):
            state[name] = generator.get_state()
    return state


def _restore_rng_state(state: Mapping[str, Any], loader: Any = None) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.random.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            print("CUDA device count changed; restored CPU RNG, CUDA RNG uses the run seed")
    if "numpy" in state:
        numpy_state = state["numpy"]
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    for name, owner in (("loader_generator", loader), ("sampler_generator", getattr(loader, "sampler", None))):
        generator = getattr(owner, "generator", None)
        if name in state and isinstance(generator, torch.Generator):
            generator.set_state(state[name].cpu())


def _save_checkpoint(
    path: Path,
    model: CabinSep,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_config: TrainingRunConfig,
    epoch_index: int,
    global_step: int,
    best_validation_loss: float,
    loss_config: LossConfig,
    next_batch_index: int = 0,
    epoch_start_rng_state: Optional[Mapping[str, Any]] = None,
    train_loader: Any = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "stage": run_config.stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": asdict(model.config),
        "training_config": asdict(run_config),
        "loss_config": asdict(loss_config),
        "epoch_index": epoch_index,
        "next_epoch_index": epoch_index,
        "next_batch_index": next_batch_index,
        "epoch_start_rng_state": epoch_start_rng_state,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng_state": _capture_rng_state(train_loader),
        "torch_version": str(torch.__version__),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _append_log(path: Path, split: str, step: int, epoch: int, values: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "step": step,
        "epoch": epoch,
        **values,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _scalar_losses(losses: Mapping[str, Tensor]) -> Dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


@torch.no_grad()
def validate(
    model: CabinSep,
    criterion: CabinSepLoss,
    loader: Iterable[Mapping[str, Any]],
    run_config: TrainingRunConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    examples = 0
    for batch in loader:
        mixture, target_speech, target_noise, lengths = adapt_batch(
            batch, run_config, model.config, device
        )
        output = forward_train_batch(model, mixture, lengths)
        losses = criterion(
            output["speech"], target_speech, output["noise"], target_noise, lengths=lengths
        )
        if not bool(torch.isfinite(losses["loss"])):
            raise FloatingPointError("Non-finite validation loss detected")
        batch_size = mixture.shape[0]
        examples += batch_size
        for name, value in _scalar_losses(losses).items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
    if examples == 0:
        raise ValueError("Validation loader yielded no batches")
    return {name: value / examples for name, value in totals.items()}


def _validate_run_config(config: TrainingRunConfig) -> None:
    if config.stage not in (1, 2):
        raise ValueError("Training stage must be 1 or 2")
    if config.resume_checkpoint and config.stage1_checkpoint:
        raise ValueError("Set resume_checkpoint or stage1_checkpoint, not both")
    if config.stage == 1 and config.stage1_checkpoint:
        raise ValueError("stage1_checkpoint is only valid when stage=2")
    for name in ("epochs", "log_every_steps", "validate_every_steps", "checkpoint_every_steps", "lr_halving_steps"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.max_steps is not None and (
        isinstance(config.max_steps, bool) or not isinstance(config.max_steps, int) or config.max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer when set")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config.gradient_clip_norm is not None and (
        not math.isfinite(config.gradient_clip_norm) or config.gradient_clip_norm <= 0
    ):
        raise ValueError("gradient_clip_norm must be finite and positive when set")
    if not isinstance(config.seed, int) or not 0 <= config.seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    if config.batch_layout.upper() not in ("BZL", "BLZ"):
        raise ValueError("batch_layout must be BZL or BLZ")
    if not config.run_name or Path(config.run_name).name != config.run_name or config.run_name in (".", ".."):
        raise ValueError("run_name must be a nonempty directory name, without parent paths")
    keys = (config.mixture_key, config.speech_key, config.noise_key)
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != 3:
        raise ValueError("mixture_key, speech_key, and noise_key must be distinct nonempty strings")
    if config.lengths_key is not None and (not isinstance(config.lengths_key, str) or not config.lengths_key):
        raise ValueError("lengths_key must be a nonempty string or None")


def run_training(
    run_config: TrainingRunConfig = TRAIN_RUN,
    *,
    train_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    validation_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    model_config: Optional[CabinSepConfig] = None,
    loss_config: Optional[LossConfig] = None,
) -> Path:
    """Train with supplied loaders or the factory in config.py; return latest.pt.

    epochs and max_steps are total limits within the current stage, including
    resumed work. Checkpoint architecture/loss settings take precedence, and
    conflicting explicitly supplied settings fail early.

    Resume restores Adam, scheduler, RNG and the next batch position. Standard
    deterministic DataLoaders can replay the epoch prefix; arbitrary external
    iterator state and persistent worker RNG state cannot be reconstructed.
    """
    _validate_run_config(run_config)
    if train_loader is None and validation_loader is not None:
        raise ValueError("Supply train_loader together with validation_loader")
    seed_everything(run_config.seed)
    device = resolve_device(run_config.device)

    resume_checkpoint: Optional[Dict[str, Any]] = None
    stage1_checkpoint: Optional[Dict[str, Any]] = None
    source_checkpoint = None
    if run_config.resume_checkpoint:
        resume_checkpoint = _load_checkpoint(_project_path(run_config.resume_checkpoint))
        source_checkpoint = resume_checkpoint
        saved_stage = resume_checkpoint.get("stage")
        if saved_stage is not None and int(saved_stage) != run_config.stage:
            raise ValueError(f"Resume checkpoint is stage {saved_stage}, requested stage is {run_config.stage}")
    elif run_config.stage == 2:
        if not run_config.stage1_checkpoint:
            raise ValueError("Stage 2 requires TRAIN_RUN.stage1_checkpoint")
        stage1_checkpoint = _load_checkpoint(_project_path(run_config.stage1_checkpoint))
        source_checkpoint = stage1_checkpoint
        saved_stage = stage1_checkpoint.get("stage")
        if saved_stage is not None and int(saved_stage) != 1:
            raise ValueError(f"Stage-2 initialization checkpoint is marked stage {saved_stage}")

    if source_checkpoint is not None:
        saved_model_config = _configuration_from_checkpoint(source_checkpoint)
        if model_config is not None and asdict(model_config) != asdict(saved_model_config):
            raise ValueError("Explicit model_config differs from checkpoint architecture")
        model_config = saved_model_config
        saved_loss_values = source_checkpoint.get("loss_config")
        if saved_loss_values is not None:
            if not isinstance(saved_loss_values, Mapping):
                raise ValueError("Checkpoint loss_config must be a mapping")
            saved_loss_config = LossConfig(**dict(saved_loss_values))
            if loss_config is not None and asdict(loss_config) != asdict(saved_loss_config):
                raise ValueError("Explicit loss_config differs from checkpoint loss settings")
            loss_config = saved_loss_config
        else:
            print("Checkpoint has no loss_config; using the supplied/default loss settings")
    model_config = (model_config or CabinSepConfig.from_variant(run_config.model_variant)).validate()
    loss_config = loss_config or LossConfig()

    run_directory = _project_path(run_config.checkpoint_dir) / run_config.run_name
    if run_directory.exists() and not run_directory.is_dir():
        raise FileExistsError(f"Run path is not a directory: {run_directory.resolve()}")
    if run_directory.exists() and any(run_directory.iterdir()):
        # Resuming within the checkpoint's own directory is deliberate. A
        # different occupied directory could silently mix unrelated experiments.
        if resume_checkpoint is None or _project_path(run_config.resume_checkpoint).resolve().parent != run_directory.resolve():
            raise FileExistsError(
                f"Run directory is not empty: {run_directory.resolve()}\n"
                "Choose a new TRAIN_RUN.run_name or resume a checkpoint from this directory."
            )

    print(f"Device: {device}")
    if device.type == "cpu":
        print("CabinSep is running on CPU; full training will be slower than on CUDA.")
    print(f"Model: CabinSep-{model_config.variant}; training stage: {run_config.stage}")
    model = CabinSep(model_config).to(device)
    criterion = CabinSepLoss(model_config.sample_rate, loss_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay)
    # [PAPER] One scheduler step per optimizer update, x0.5 every 20,000 updates.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=run_config.lr_halving_steps, gamma=0.5)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_validation_loss = float("inf")
    resume_rng = None
    resume_epoch_rng = None
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        for required in ("optimizer_state_dict", "scheduler_state_dict"):
            if required not in resume_checkpoint:
                raise ValueError(f"Resume checkpoint is missing {required}")
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = int(resume_checkpoint.get("next_epoch_index", resume_checkpoint.get("epoch_index", 0)))
        start_batch = int(resume_checkpoint.get("next_batch_index", 0))
        global_step = int(resume_checkpoint.get("global_step", 0))
        if min(start_epoch, start_batch, global_step) < 0:
            raise ValueError("Checkpoint epoch, batch, and step counters must be nonnegative")
        best_validation_loss = float(resume_checkpoint.get("best_validation_loss", float("inf")))
        if math.isnan(best_validation_loss):
            raise ValueError("Checkpoint best_validation_loss is NaN")
        resume_rng = resume_checkpoint.get("rng_state", {
            "python": resume_checkpoint.get("python_rng_state", random.getstate()),
            "torch": resume_checkpoint.get("torch_rng_state", torch.random.get_rng_state()),
            "cuda": resume_checkpoint.get("cuda_rng_states", []),
        })
        resume_epoch_rng = resume_checkpoint.get("epoch_start_rng_state")
        if "next_epoch_index" not in resume_checkpoint:
            print("Legacy checkpoint has no batch position; resume starts at its saved epoch boundary")
        scheduled_lrs = scheduler.get_last_lr()
        optimizer_lrs = [group["lr"] for group in optimizer.param_groups]
        if len(scheduled_lrs) != len(optimizer_lrs) or any(
            abs(a - b) > 1.0e-15 for a, b in zip(scheduled_lrs, optimizer_lrs)
        ):
            raise ValueError("Checkpoint optimizer LR and scheduler state are inconsistent")
        print(f"Resumed optimizer/scheduler at step {global_step}, epoch {start_epoch + 1}, next batch {start_batch + 1}")
        if start_epoch >= run_config.epochs or (
            run_config.max_steps is not None and global_step >= run_config.max_steps
        ):
            print("Requested total epoch/step limit is already reached; no optimizer update performed.")
            return _project_path(run_config.resume_checkpoint).resolve()
    elif stage1_checkpoint is not None:
        model.load_state_dict(stage1_checkpoint["model_state_dict"], strict=True)
        # [ASSUMPTION] The paper does not say whether Adam moments carry over.
        print("Loaded Stage-1 model weights; Stage-2 optimizer and step count start fresh")

    if train_loader is None:
        train_loader, validation_loader = build_external_loaders(run_config, model_config)
    if train_loader is None or not hasattr(train_loader, "__iter__"):
        raise TypeError("The training loader must be an iterable of batch mappings")
    if isinstance(train_loader, Iterator):
        raise TypeError("train_loader must be reusable across epochs; supply a DataLoader/list, not a one-shot iterator")
    if validation_loader is not None and (
        not hasattr(validation_loader, "__iter__") or isinstance(validation_loader, Iterator)
    ):
        raise TypeError("validation_loader must be reusable, such as a DataLoader/list")
    if resume_rng is not None:
        _restore_rng_state(resume_rng, train_loader)

    log_path = run_directory / "losses.jsonl"
    latest_path = run_directory / "latest.pt"
    best_path = run_directory / "best.pt"
    if resume_checkpoint is not None and validation_loader is not None and math.isfinite(best_validation_loss):
        source_best_path = _project_path(run_config.resume_checkpoint).resolve().parent / "best.pt"
        historical_best = _load_checkpoint(source_best_path) if source_best_path.is_file() else None
        if historical_best is not None and (
            historical_best.get("best_validation_loss") == best_validation_loss
            and int(historical_best.get("global_step", global_step + 1)) <= global_step
            and historical_best.get("model_config") == resume_checkpoint.get("model_config")
        ):
            if source_best_path != best_path.resolve():
                best_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_best_path, best_path)
        else:
            print("Historical best model weights are unavailable for this resume point; best selection restarts.")
            best_validation_loss = float("inf")
    epoch_rng: Optional[Mapping[str, Any]] = None
    last_validation_step = -1

    def save(path: Path, next_epoch: int, next_batch: int) -> None:
        _save_checkpoint(
            path, model, optimizer, scheduler, run_config, next_epoch, global_step,
            best_validation_loss, loss_config, next_batch,
            epoch_rng if next_batch else None, train_loader,
        )

    def evaluate(next_epoch: int, next_batch: int, display_epoch: int) -> None:
        nonlocal best_validation_loss, last_validation_step
        if validation_loader is None or last_validation_step == global_step:
            return
        # Validation must not change the training RNG sequence (some external
        # validation transforms are random even while the model is in eval).
        validation_rng = _capture_rng_state(train_loader)
        was_training = model.training
        try:
            values = validate(model, criterion, validation_loader, run_config, device)
        finally:
            _restore_rng_state(validation_rng, train_loader)
            model.train(was_training)
        last_validation_step = global_step
        _append_log(log_path, "validation", global_step, display_epoch, values)
        print(f"validation step={global_step} loss={values['loss']:.6f}")
        if values["loss"] < best_validation_loss:
            best_validation_loss = values["loss"]
            save(best_path, next_epoch, next_batch)

    for epoch_index in range(start_epoch, run_config.epochs):
        # DistributedSampler.set_epoch is also useful for deterministic replay.
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_index)
        batch_offset = start_batch if epoch_index == start_epoch else 0
        if batch_offset and resume_epoch_rng is not None:
            _restore_rng_state(resume_epoch_rng, train_loader)
        epoch_rng = _capture_rng_state(train_loader)
        iterator = iter(train_loader)
        for _ in range(batch_offset):
            try:
                next(iterator)
            except StopIteration as error:
                raise ValueError("Resumed loader has fewer batches than the checkpoint's batch position") from error
        if batch_offset and resume_rng is not None:
            _restore_rng_state(resume_rng, train_loader)
        model.train()
        batches_seen = batch_offset
        stop_requested = False

        for batch in iterator:
            mixture, target_speech, target_noise, lengths = adapt_batch(batch, run_config, model_config, device)
            optimizer.zero_grad(set_to_none=True)
            output = forward_train_batch(model, mixture, lengths)
            losses = criterion(output["speech"], target_speech, output["noise"], target_noise, lengths=lengths)
            total_loss = losses["loss"]
            if not bool(torch.isfinite(total_loss)):
                diagnostic_path = run_directory / f"nonfinite_step_{global_step:08d}.pt"
                save(diagnostic_path, epoch_index, batches_seen)
                raise FloatingPointError(f"Non-finite training loss at step {global_step}; saved {diagnostic_path}")
            total_loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                diagnostic_path = run_directory / f"nonfinite_gradient_step_{global_step:08d}.pt"
                save(diagnostic_path, epoch_index, batches_seen)
                raise FloatingPointError(f"Non-finite gradient at step {global_step}; saved {diagnostic_path}")
            if run_config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            global_step += 1
            batches_seen += 1
            scalars = _scalar_losses(losses)
            if global_step == 1 or global_step % run_config.log_every_steps == 0:
                scalars["learning_rate"] = optimizer.param_groups[0]["lr"]
                _append_log(log_path, "train", global_step, epoch_index + 1, scalars)
                print(f"epoch={epoch_index + 1} step={global_step} loss={scalars['loss']:.6f} lr={scalars['learning_rate']:.3e}")

            # Evaluate first so periodic checkpoints carry the current best score.
            if global_step % run_config.validate_every_steps == 0:
                evaluate(epoch_index, batches_seen, epoch_index + 1)
            if global_step % run_config.checkpoint_every_steps == 0:
                save(run_directory / f"step_{global_step:08d}.pt", epoch_index, batches_seen)
                save(latest_path, epoch_index, batches_seen)
            if run_config.max_steps is not None and global_step >= run_config.max_steps:
                stop_requested = True
                break

        if batches_seen == 0:
            raise ValueError("Training loader yielded no batches")
        # A limit reached inside an epoch retains its batch position; on resume
        # the remaining batches are consumed before advancing the epoch counter.
        next_epoch = epoch_index if stop_requested else epoch_index + 1
        next_batch = batches_seen if stop_requested else 0
        evaluate(next_epoch, next_batch, epoch_index + 1)
        save(latest_path, next_epoch, next_batch)
        if validation_loader is None:
            # Without validation, best.pt is a convenient alias of latest.pt.
            save(best_path, next_epoch, next_batch)
        if stop_requested:
            break

    print(f"Training complete at step {global_step}. Latest checkpoint: {latest_path.resolve()}")
    return latest_path.resolve()


def main() -> None:
    # A checkpoint is authoritative on resume/Stage 2. Fresh runs use the
    # editable architecture and loss objects at the bottom of config.py.
    if TRAIN_RUN.resume_checkpoint or TRAIN_RUN.stage1_checkpoint:
        run_training(TRAIN_RUN)
    else:
        run_training(
            replace(TRAIN_RUN, model_variant=TRAIN_MODEL.variant),
            model_config=TRAIN_MODEL,
            loss_config=TRAIN_LOSS,
        )


if __name__ == "__main__":
    # Required on Windows when the external DataLoader uses worker processes.
    main()
