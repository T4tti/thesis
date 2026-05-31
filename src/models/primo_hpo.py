"""PriMO/NePS HPO utilities for the Transformer-LSTM notebook.

The notebook owns the dataset, model class, and loss class. This module owns
the search-space definition, trial orchestration, fallback optimizer, and
artifact extraction so Cell 24 can stay reproducible on Kaggle.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable

try:
    import neps  # type: ignore[import-not-found]

    _NEPS_AVAILABLE = True
except ImportError:
    neps = None  # type: ignore[assignment]
    _NEPS_AVAILABLE = False


PRIMO_EXPERT_PRIOR: dict[str, Any] = {
    "hidden_size": 64,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 1,
    "dropout": 0.28,
    "batch_size": 64,
    "lr": 2.5e-4,
    "max_lr": 4.0e-4,
    "weight_decay": 4e-3,
    "focal_gamma": 1.2,
    "focal_weight": 0.05,
    "ordinal_alpha": 0.04,
    "label_smoothing": 0.015,
    "context_dropout_prob": 0.08,
    "train_input_noise_std": 0.005,
    "train_channel_dropout_prob": 0.02,
    "train_timestep_dropout_prob": 0.02,
    "loss_gap_penalty_weight": 0.20,
    "sector_emb_dim": 16,
    "ticker_emb_dim": 16,
    "company_emb_dim": 16,
    "max_relative_position": 8,
}

_HEAD_CHOICES = (1, 2, 4, 8)
_BATCH_SIZE_CHOICES = (32, 48, 64, 96, 128)
_PRIMO_CONTEXT: dict[str, Any] = {}
_TRIAL_ROWS: list[dict[str, Any]] = []


def snap_n_heads(d_model: int, n_heads: int) -> int:
    """Return the nearest valid attention-head count for d_model."""
    d_model = int(d_model)
    n_heads = int(n_heads)
    if d_model < n_heads:
        return 1
    valid = [h for h in _HEAD_CHOICES if h <= d_model and d_model % h == 0]
    if not valid:
        return 1
    return int(min(valid, key=lambda h: (abs(h - n_heads), -h)))


def snap_batch_size(batch_size: int) -> int:
    """Return the nearest Kaggle-safe batch size candidate."""
    batch_size = int(batch_size)
    return int(min(_BATCH_SIZE_CHOICES, key=lambda b: (abs(b - batch_size), b)))


def _make_neps_dimension(kind: str, **kwargs: Any) -> Any:
    """Create a NePS dimension while tolerating minor API differences."""
    if neps is None:
        raise ImportError("neps is not installed")
    cls = getattr(neps, kind)
    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop("default", None)
        try:
            return cls(**kwargs)
        except TypeError:
            kwargs.pop("is_fidelity", None)
            return cls(**kwargs)


def build_pipeline_space() -> dict[str, Any]:
    """Create the NePS pipeline space with thesis expert priors."""
    if neps is None:
        raise ImportError("NePS is not installed; use fallback PriMO instead.")
    return {
        "hidden_size": _make_neps_dimension(
            "Integer", lower=32, upper=256, default=PRIMO_EXPERT_PRIOR["hidden_size"], log=True
        ),
        "d_model": _make_neps_dimension(
            "Integer", lower=32, upper=256, default=PRIMO_EXPERT_PRIOR["d_model"], log=True
        ),
        "n_heads": _make_neps_dimension(
            "Categorical", choices=list(_HEAD_CHOICES), default=PRIMO_EXPERT_PRIOR["n_heads"]
        ),
        "n_layers": _make_neps_dimension("Integer", lower=1, upper=4, default=PRIMO_EXPERT_PRIOR["n_layers"]),
        "dropout": _make_neps_dimension("Float", lower=0.10, upper=0.40, default=PRIMO_EXPERT_PRIOR["dropout"]),
        "batch_size": _make_neps_dimension(
            "Categorical", choices=list(_BATCH_SIZE_CHOICES), default=PRIMO_EXPERT_PRIOR["batch_size"]
        ),
        "lr": _make_neps_dimension("Float", lower=1e-5, upper=1e-2, default=PRIMO_EXPERT_PRIOR["lr"], log=True),
        "max_lr": _make_neps_dimension("Float", lower=5e-5, upper=2e-3, default=PRIMO_EXPERT_PRIOR["max_lr"], log=True),
        "weight_decay": _make_neps_dimension(
            "Float", lower=1e-4, upper=1.2e-2, default=PRIMO_EXPERT_PRIOR["weight_decay"], log=True
        ),
        "focal_gamma": _make_neps_dimension("Float", lower=0.8, upper=1.6, default=PRIMO_EXPERT_PRIOR["focal_gamma"]),
        "focal_weight": _make_neps_dimension(
            "Float", lower=0.0, upper=0.12, default=PRIMO_EXPERT_PRIOR["focal_weight"]
        ),
        "ordinal_alpha": _make_neps_dimension(
            "Float", lower=0.0, upper=0.10, default=PRIMO_EXPERT_PRIOR["ordinal_alpha"]
        ),
        "label_smoothing": _make_neps_dimension(
            "Float", lower=0.0, upper=0.035, default=PRIMO_EXPERT_PRIOR["label_smoothing"]
        ),
        "context_dropout_prob": _make_neps_dimension(
            "Float", lower=0.0, upper=0.18, default=PRIMO_EXPERT_PRIOR["context_dropout_prob"]
        ),
        "train_input_noise_std": _make_neps_dimension(
            "Float", lower=0.0, upper=0.020, default=PRIMO_EXPERT_PRIOR["train_input_noise_std"]
        ),
        "train_channel_dropout_prob": _make_neps_dimension(
            "Float", lower=0.0, upper=0.060, default=PRIMO_EXPERT_PRIOR["train_channel_dropout_prob"]
        ),
        "train_timestep_dropout_prob": _make_neps_dimension(
            "Float", lower=0.0, upper=0.060, default=PRIMO_EXPERT_PRIOR["train_timestep_dropout_prob"]
        ),
        "loss_gap_penalty_weight": _make_neps_dimension(
            "Float", lower=0.05, upper=0.35, default=PRIMO_EXPERT_PRIOR["loss_gap_penalty_weight"]
        ),
        "sector_emb_dim": _make_neps_dimension(
            "Categorical", choices=[8, 16, 32], default=PRIMO_EXPERT_PRIOR["sector_emb_dim"]
        ),
        "ticker_emb_dim": _make_neps_dimension(
            "Categorical", choices=[4, 8, 16, 32], default=PRIMO_EXPERT_PRIOR["ticker_emb_dim"]
        ),
        "company_emb_dim": _make_neps_dimension(
            "Categorical", choices=[4, 8, 16, 32], default=PRIMO_EXPERT_PRIOR["company_emb_dim"]
        ),
        "max_relative_position": _make_neps_dimension(
            "Integer", lower=4, upper=8, default=PRIMO_EXPERT_PRIOR["max_relative_position"]
        ),
        "epochs": _make_neps_dimension("Integer", lower=5, upper=20, default=10, is_fidelity=True),
    }


def configure_primo_context(**kwargs: Any) -> None:
    """Store notebook-owned objects used by NePS trial callbacks."""
    _PRIMO_CONTEXT.clear()
    _PRIMO_CONTEXT.update(kwargs)


def _subsample_dataset(dataset: Any, fraction: float, seed: int = 42) -> Any:
    """Sample a deterministic fraction of a torch Dataset."""
    if fraction >= 1.0:
        return dataset
    from torch.utils.data import Subset

    n = max(1, int(len(dataset) * float(fraction)))
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n)
    return Subset(dataset, indices)


def _unpack_batch(batch: Any, device: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Move the notebook dataset batch to device.

    Current Transformer-LSTM batches are:
    (X, last_y, sector_id, ticker_id, company_id, y, row_id).
    The older four-field format is kept as a fallback for exported scripts.
    """
    if len(batch) >= 7:
        x, last_y, sector_id, ticker_id, company_id, y, _row_id = batch[:7]
    elif len(batch) == 6:
        x, y, sector_id, ticker_id, company_id, last_y = batch
    elif len(batch) == 4:
        x, last_y, sector_id, y = batch
        ticker_id = company_id = None
    else:
        raise ValueError(f"Unsupported batch structure with {len(batch)} fields")

    def _move(value: Any) -> Any:
        if value is None:
            return None
        return value.to(device, non_blocking=True)

    return _move(x), _move(y), _move(sector_id), _move(ticker_id), _move(company_id), _move(last_y)


def _call_model(model: Any, x: Any, sector_id: Any, ticker_id: Any, company_id: Any, last_y: Any) -> Any:
    """Call the notebook model with its current argument order."""
    return model(x, last_y, sector_id, ticker_id, company_id)


def _append_trial_row(row: dict[str, Any]) -> None:
    _TRIAL_ROWS.append(row)
    artifact_dir = _PRIMO_CONTEXT.get("artifact_dir")
    if artifact_dir is None:
        return
    path = Path(artifact_dir) / "transformer_primo_trials.csv"
    fieldnames = [
        "trial_id",
        "epochs",
        "composite_score",
        "val_loss",
        "val_accuracy",
        "f1_weighted",
        "f1_macro",
        "qwk",
        "hidden_size",
        "d_model",
        "n_heads",
        "n_layers",
        "dropout",
        "batch_size",
        "lr",
        "max_lr",
        "weight_decay",
        "focal_weight",
        "context_dropout_prob",
        "train_input_noise_std",
        "train_channel_dropout_prob",
        "train_timestep_dropout_prob",
        "loss_gap_penalty_weight",
        "train_loss",
        "loss_gap",
        "chgacc",
        "persistence_rate",
        "elapsed_seconds",
    ]
    extras = [k for k in row if k not in fieldnames]
    fieldnames.extend(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_TRIAL_ROWS)


def _write_top_configs(artifact_dir: str | Path, top_k: int = 5) -> None:
    """Save a compact top-config table sorted by the aligned PriMO score."""
    if not _TRIAL_ROWS:
        return
    rows = sorted(
        _TRIAL_ROWS, key=lambda row: float(row.get("aligned_score", row.get("composite_score", 0.0))), reverse=True
    )
    path = Path(artifact_dir) / "transformer_primo_top_configs.csv"
    fieldnames = []
    for row in rows[:top_k]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[:top_k])


def _rerun_top_configs(root_directory: str | Path, top_k: int = 3, epochs: int = 40) -> None:
    """Second-stage rerun of the best proxy configs on full train fraction."""
    if not _TRIAL_ROWS or top_k <= 0:
        return
    root = Path(root_directory)
    rows = sorted(
        _TRIAL_ROWS, key=lambda row: float(row.get("aligned_score", row.get("composite_score", 0.0))), reverse=True
    )
    original_train_fraction = _PRIMO_CONTEXT.get("train_fraction", 0.50)
    _PRIMO_CONTEXT["train_fraction"] = 1.0
    try:
        for rank, row in enumerate(rows[:top_k], start=1):
            cfg = {k: row[k] for k in PRIMO_EXPERT_PRIOR if k in row}
            print(f"[PriMO-NePS] Stage-2 rerun top-{rank}/{top_k} for {epochs} epochs")
            evaluate_pipeline(
                pipeline_directory=str(root / f"stage2_top_{rank:02d}"),
                previous_pipeline_directory=None,
                epochs=int(epochs),
                **cfg,
            )
    finally:
        _PRIMO_CONTEXT["train_fraction"] = original_train_fraction


def _apply_sequence_regularization(x_batch: Any, config: dict[str, Any], torch_module: Any) -> Any:
    """Mirror Cell 27 input noise/channel/timestep dropout inside HPO trials."""
    x_aug = x_batch
    noise_std = float(config.get("train_input_noise_std", 0.0))
    channel_dropout = float(config.get("train_channel_dropout_prob", 0.0))
    timestep_dropout = float(config.get("train_timestep_dropout_prob", 0.0))
    if noise_std > 0:
        x_aug = x_aug + torch_module.randn_like(x_aug) * noise_std
    if channel_dropout > 0:
        channel_mask = torch_module.rand((x_aug.shape[0], 1, x_aug.shape[2]), device=x_aug.device) < channel_dropout
        x_aug = x_aug.masked_fill(channel_mask, 0.0)
    if timestep_dropout > 0 and x_aug.shape[1] > 1:
        time_mask = torch_module.rand((x_aug.shape[0], x_aug.shape[1], 1), device=x_aug.device) < timestep_dropout
        time_mask[:, -1, :] = False
        x_aug = x_aug.masked_fill(time_mask, 0.0)
    return x_aug


def _prepare_one_cycle_resume_groups(
    optimizer: Any,
    *,
    max_lr: float,
    base_lr: float,
    final_div_factor: float,
    base_momentum: float = 0.85,
    max_momentum: float = 0.95,
) -> None:
    """Fill OneCycleLR param-group metadata when resuming old checkpoints."""
    div_factor = max(float(max_lr) / max(float(base_lr), 1e-9), 1.0)
    initial_lr = float(max_lr) / div_factor
    min_lr = initial_lr / float(final_div_factor)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", initial_lr)
        group.setdefault("max_lr", float(max_lr))
        group.setdefault("min_lr", min_lr)
        if "betas" in group or "momentum" in group:
            group.setdefault("base_momentum", float(base_momentum))
            group.setdefault("max_momentum", float(max_momentum))


def _build_one_cycle_scheduler(
    optimizer: Any,
    torch_module: Any,
    config: dict[str, Any],
    *,
    steps_per_epoch: int,
    total_epochs: int,
    start_epoch: int = 0,
) -> Any:
    """Build OneCycleLR aligned to the target fidelity and completed warm-start steps."""
    total_epochs = max(int(total_epochs), 1)
    steps_per_epoch = max(int(steps_per_epoch), 1)
    max_lr = float(config["max_lr"])
    base_lr = float(config["lr"])
    div_factor = max(max_lr / max(base_lr, 1e-9), 1.0)
    final_div_factor = 100.0
    completed_steps = min(max(int(start_epoch), 0) * steps_per_epoch, total_epochs * steps_per_epoch)
    last_epoch = completed_steps - 1 if completed_steps > 0 else -1
    if last_epoch >= 0:
        _prepare_one_cycle_resume_groups(
            optimizer,
            max_lr=max_lr,
            base_lr=base_lr,
            final_div_factor=final_div_factor,
        )
    return torch_module.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        pct_start=min(0.30, 2.0 / total_epochs),
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        last_epoch=last_epoch,
    )


def _result_from_exception(exc: BaseException, elapsed: float = 0.0) -> dict[str, Any]:
    print(f"  [Trial FAILED] {type(exc).__name__}: {exc}")
    traceback.print_exc()
    return {
        "loss": 1.0,
        "cost": float(elapsed),
        "info_dict": {
            "error": str(exc),
            "val_accuracy": 0.0,
            "composite_score": 0.0,
            "obj_1_neg_composite": 1.0,
            "obj_2_val_loss": float("inf"),
        },
    }


def evaluate_pipeline(
    pipeline_directory: str,
    previous_pipeline_directory: str | None = None,
    *,
    hidden_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    batch_size: int,
    lr: float,
    max_lr: float,
    weight_decay: float,
    focal_gamma: float,
    focal_weight: float,
    ordinal_alpha: float,
    label_smoothing: float,
    context_dropout_prob: float,
    train_input_noise_std: float,
    train_channel_dropout_prob: float,
    train_timestep_dropout_prob: float,
    loss_gap_penalty_weight: float,
    sector_emb_dim: int,
    ticker_emb_dim: int,
    company_emb_dim: int,
    max_relative_position: int,
    epochs: int,
) -> dict[str, Any]:
    """Evaluate one PriMO/NePS trial and return a NePS-compatible result."""
    t_start = time.time()
    try:
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
        from torch.utils.data import DataLoader

        required = ("train_ds", "val_ds", "model_factory", "criterion_factory", "device")
        missing = [k for k in required if k not in _PRIMO_CONTEXT]
        if missing:
            raise RuntimeError(f"PriMO context is missing keys: {missing}")

        device = _PRIMO_CONTEXT["device"]
        use_cuda = bool(getattr(device, "type", str(device)) == "cuda")
        trial_batch_size = snap_batch_size(int(batch_size or _PRIMO_CONTEXT.get("batch_size", 64)))
        train_fraction = float(_PRIMO_CONTEXT.get("train_fraction", 0.50))
        val_fraction = float(_PRIMO_CONTEXT.get("val_fraction", 1.00))
        random_state = int(_PRIMO_CONTEXT.get("random_state", 42))
        trial_id = Path(pipeline_directory).name

        n_heads = snap_n_heads(d_model, n_heads)
        config = {
            "hidden_size": int(hidden_size),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "dropout": float(dropout),
            "batch_size": int(trial_batch_size),
            "lr": float(lr),
            "max_lr": max(float(max_lr), float(lr) * 1.05),
            "weight_decay": float(weight_decay),
            "focal_gamma": float(focal_gamma),
            "focal_weight": float(focal_weight),
            "ordinal_alpha": float(ordinal_alpha),
            "label_smoothing": float(label_smoothing),
            "context_dropout_prob": float(context_dropout_prob),
            "train_input_noise_std": float(train_input_noise_std),
            "train_channel_dropout_prob": float(train_channel_dropout_prob),
            "train_timestep_dropout_prob": float(train_timestep_dropout_prob),
            "loss_gap_penalty_weight": float(loss_gap_penalty_weight),
            "sector_emb_dim": int(sector_emb_dim),
            "ticker_emb_dim": int(ticker_emb_dim),
            "company_emb_dim": int(company_emb_dim),
            "max_relative_position": int(max_relative_position),
        }
        # Keep the HPO objective aligned with final training without inflating the
        # reported loss through excessive auxiliary regularization.
        config["dropout"] = float(np.clip(float(config["dropout"]), 0.10, 0.40))
        config["weight_decay"] = float(np.clip(float(config["weight_decay"]), 1e-4, 1.2e-2))
        config["focal_gamma"] = float(np.clip(float(config["focal_gamma"]), 0.8, 1.6))
        config["focal_weight"] = float(np.clip(float(config["focal_weight"]), 0.0, 0.12))
        config["ordinal_alpha"] = float(np.clip(float(config["ordinal_alpha"]), 0.0, 0.10))
        config["label_smoothing"] = float(np.clip(float(config["label_smoothing"]), 0.0, 0.035))

        train_frac_eff = min(train_fraction, 0.50) if int(epochs) <= 10 else train_fraction
        train_subset = _subsample_dataset(_PRIMO_CONTEXT["train_ds"], train_frac_eff, seed=random_state)
        val_subset = _subsample_dataset(_PRIMO_CONTEXT["val_ds"], val_fraction, seed=random_state)
        train_loader = DataLoader(
            train_subset,
            batch_size=trial_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=max(trial_batch_size * 2, 1),
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )

        model = _PRIMO_CONTEXT["model_factory"](config)
        criterion = _PRIMO_CONTEXT["criterion_factory"](config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            eps=1e-8,
        )

        start_epoch = 0
        resumed_train_loss: float | None = None
        total_epochs = max(int(epochs), 1)
        steps_per_epoch = max(1, len(train_loader))
        pipeline_dir = Path(pipeline_directory)
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = pipeline_dir / "checkpoint.pt"
        if previous_pipeline_directory is not None:
            prev_ckpt = Path(previous_pipeline_directory) / "checkpoint.pt"
            if prev_ckpt.exists():
                try:
                    state = torch.load(prev_ckpt, map_location=device)
                    model.load_state_dict(state["model"])
                    optimizer.load_state_dict(state["optimizer"])
                    start_epoch = min(max(int(state.get("epoch", 0)), 0), total_epochs)
                    if "train_loss" in state:
                        resumed_train_loss = float(state["train_loss"])
                    resume_step = start_epoch * steps_per_epoch
                    print(
                        f"  [Warm-start] Resumed from epoch {start_epoch}; "
                        f"OneCycleLR continues at step {resume_step}/{total_epochs * steps_per_epoch}"
                    )
                except Exception as exc:
                    print(f"  [Warm-start WARN] Failed to load checkpoint: {exc}")
                    start_epoch = 0
        scheduler = _build_one_cycle_scheduler(
            optimizer,
            torch,
            config,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            start_epoch=start_epoch,
        )

        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        train_losses: list[float] = []
        for _epoch in range(start_epoch, int(epochs)):
            model.train()
            for batch in train_loader:
                x, y, sector_id, ticker_id, company_id, last_y = _unpack_batch(batch, device)
                x = _apply_sequence_regularization(x, config, torch)
                context_dropout = float(config.get("context_dropout_prob", 0.0))
                if context_dropout > 0:
                    drop_mask = torch.rand(last_y.shape, device=device) < context_dropout
                    last_y = last_y.masked_fill(drop_mask, 0)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits = _call_model(model, x, sector_id, ticker_id, company_id, last_y)
                    loss = criterion(logits, y)
                train_losses.append(float(loss.detach().cpu().item()))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        model.eval()
        val_losses: list[float] = []
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_last_y: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                x, y, sector_id, ticker_id, company_id, last_y = _unpack_batch(batch, device)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits = _call_model(model, x, sector_id, ticker_id, company_id, last_y)
                    loss = criterion(logits, y)
                val_losses.append(float(loss.item()))
                all_preds.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())
                all_labels.extend(y.detach().cpu().numpy().tolist())
                all_last_y.extend(last_y.detach().cpu().numpy().tolist())

        train_loss = (
            float(np.mean(train_losses))
            if train_losses
            else (float(resumed_train_loss) if resumed_train_loss is not None else float("inf"))
        )
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_acc = float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0
        f1_weighted = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)) if all_labels else 0.0
        f1_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0)) if all_labels else 0.0
        try:
            qwk = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))
            if not np.isfinite(qwk):
                qwk = 0.0
        except Exception:
            qwk = 0.0
        labels_np = np.asarray(all_labels, dtype=int)
        preds_np = np.asarray(all_preds, dtype=int)
        last_y_np = np.asarray(all_last_y, dtype=int)
        change_mask = labels_np != last_y_np
        if change_mask.any():
            chgacc = float(np.mean(preds_np[change_mask] == labels_np[change_mask]))
        else:
            chgacc = val_acc
        persistence_rate = float(np.mean(preds_np == last_y_np)) if len(last_y_np) else 0.0
        loss_gap = (
            max(0.0, float(val_loss) - float(train_loss)) if np.isfinite(val_loss) and np.isfinite(train_loss) else 0.0
        )
        aligned_score = (
            0.30 * f1_weighted
            + 0.20 * f1_macro
            + 0.20 * qwk
            + 0.15 * chgacc
            + 0.10 * val_acc
            - float(config["loss_gap_penalty_weight"]) * loss_gap
            - 0.05 * persistence_rate
        )
        composite = aligned_score
        obj_1 = float(1.0 - aligned_score)
        obj_2 = float(val_loss)

        torch.save(
            {
                "epoch": int(epochs),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scheduler_name": "OneCycleLR",
                "scheduler_steps_per_epoch": int(steps_per_epoch),
                "scheduler_total_epochs": int(total_epochs),
                "train_loss": train_loss,
                "val_loss": obj_2,
                "config": config,
                "obj_1": obj_1,
                "obj_2": obj_2,
            },
            ckpt_path,
        )

        elapsed = float(time.time() - t_start)
        info = {
            "obj_1_neg_composite": obj_1,
            "obj_2_val_loss": obj_2,
            "train_loss": train_loss,
            "loss_gap": loss_gap,
            "val_accuracy": val_acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "qwk": qwk,
            "chgacc": chgacc,
            "persistence_rate": persistence_rate,
            "composite_score": float(composite),
            "aligned_score": float(aligned_score),
            "epochs_trained": int(epochs),
            "config": config,
        }
        _append_trial_row(
            {
                "trial_id": trial_id,
                "epochs": int(epochs),
                "composite_score": float(composite),
                "aligned_score": float(aligned_score),
                "train_loss": train_loss,
                "val_loss": obj_2,
                "loss_gap": loss_gap,
                "val_accuracy": val_acc,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
                "qwk": qwk,
                "chgacc": chgacc,
                "persistence_rate": persistence_rate,
                "elapsed_seconds": elapsed,
                **config,
            }
        )
        print(
            f"[NePS] Trial {trial_id} | epochs={int(epochs):02d} | "
            f"score={aligned_score:.4f} | chgacc={chgacc:.4f} | val_loss={val_loss:.4f}"
        )
        del model, optimizer, criterion, scaler
        if use_cuda:
            torch.cuda.empty_cache()
        return {"loss": obj_1, "cost": elapsed, "info_dict": info}
    except Exception as exc:
        return _result_from_exception(exc, elapsed=time.time() - t_start)


def _extract_best_config_from_neps(root_directory: str | Path) -> dict[str, Any]:
    """Read NePS/fallback artifacts and return the config with lowest primary loss."""
    root = Path(root_directory)
    best_loss = float("inf")
    best_config: dict[str, Any] | None = None

    trials_csv = root.parent / "transformer_primo_trials.csv"
    if trials_csv.exists():
        with trials_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    score_key = "aligned_score" if row.get("aligned_score") not in (None, "") else "composite_score"
                    loss = 1.0 - float(row[score_key])
                except Exception:
                    continue
                if loss < best_loss:
                    best_loss = loss
                    best_config = {k: _coerce_config_value(k, row[k]) for k in PRIMO_EXPERT_PRIOR if k in row}

    results_dir = root / "results"
    if results_dir.exists():
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            yaml = None  # type: ignore[assignment]
        if yaml is not None:
            for trial_dir in sorted(results_dir.iterdir()):
                result_file = trial_dir / "result.yaml"
                config_file = trial_dir / "config.yaml"
                if not result_file.exists() or not config_file.exists():
                    continue
                try:
                    result = yaml.safe_load(result_file.read_text(encoding="utf-8")) or {}
                    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                    trial_loss = result.get("loss", result.get("objective_to_minimize", [float("inf")])[0])
                    if isinstance(trial_loss, (int, float)) and float(trial_loss) < best_loss:
                        best_loss = float(trial_loss)
                        best_config = dict(config)
                except Exception:
                    continue

    if best_config is None:
        print("[WARN] No completed PriMO trials found. Returning expert prior.")
        return dict(PRIMO_EXPERT_PRIOR)

    best_config.pop("epochs", None)
    best_config["n_heads"] = snap_n_heads(int(best_config["d_model"]), int(best_config["n_heads"]))
    if "batch_size" in best_config:
        best_config["batch_size"] = snap_batch_size(int(best_config["batch_size"]))
    print(f"[NePS] Best config found: 1-composite={best_loss:.4f}, composite={1.0 - best_loss:.4f}")
    return best_config


def _coerce_config_value(key: str, value: Any) -> Any:
    prior = PRIMO_EXPERT_PRIOR[key]
    if isinstance(prior, bool):
        return str(value).lower() in {"1", "true", "yes"}
    if isinstance(prior, int):
        return int(float(value))
    if isinstance(prior, float):
        return float(value)
    return value


def _run_fallback_primo(
    pipeline_space_defaults: dict[str, Any],
    max_evaluations_total: int,
    root_directory: str | Path,
) -> dict[str, Any]:
    """Prior-guided single-process fallback when NePS is unavailable."""
    root = Path(root_directory)
    root.mkdir(parents=True, exist_ok=True)
    best_config = dict(pipeline_space_defaults)
    best_loss = float("inf")
    rng = random.Random(int(_PRIMO_CONTEXT.get("random_state", 42)))
    ranges = {
        "hidden_size": (32, 256),
        "d_model": (32, 256),
        "n_layers": (1, 4),
        "dropout": (0.10, 0.40),
        "lr": (-5, -2),
        "max_lr": (-4.3, -2.7),
        "weight_decay": (-4, -1.92),
        "focal_gamma": (0.8, 1.6),
        "focal_weight": (0.0, 0.12),
        "ordinal_alpha": (0.0, 0.10),
        "label_smoothing": (0.0, 0.035),
        "context_dropout_prob": (0.0, 0.18),
        "train_input_noise_std": (0.0, 0.020),
        "train_channel_dropout_prob": (0.0, 0.060),
        "train_timestep_dropout_prob": (0.0, 0.060),
        "loss_gap_penalty_weight": (0.05, 0.35),
    }

    for trial_idx in range(int(max_evaluations_total)):
        print(f"  [Fallback PriMO] Trial {trial_idx + 1}/{max_evaluations_total}")
        cfg = dict(best_config) if trial_idx == 0 else _perturb_config(best_config, ranges, sigma=0.15, rng=rng)
        fidelity_epochs = min(5 + trial_idx, 20)
        result = evaluate_pipeline(
            pipeline_directory=str(root / f"trial_{trial_idx:04d}"),
            previous_pipeline_directory=None,
            epochs=fidelity_epochs,
            **cfg,
        )
        if float(result["loss"]) < best_loss:
            best_loss = float(result["loss"])
            best_config = dict(cfg)
            info = result.get("info_dict", {})
            print(
                f"    [New best] composite={1.0 - best_loss:.4f}, "
                f"val_loss={float(info.get('obj_2_val_loss', float('inf'))):.4f}"
            )

    best_config.pop("epochs", None)
    best_config["n_heads"] = snap_n_heads(int(best_config["d_model"]), int(best_config["n_heads"]))
    return best_config


def _perturb_config(
    config: dict[str, Any], ranges: dict[str, tuple[float, float]], sigma: float, rng: random.Random
) -> dict[str, Any]:
    """Gaussian perturbation around the current best config."""
    cfg = copy.deepcopy(config)
    for key, (lo, hi) in ranges.items():
        if key not in cfg:
            continue
        value = cfg[key]
        if key in {"lr", "max_lr", "weight_decay"}:
            log_value = math.log10(float(value))
            perturbed = max(lo, min(hi, log_value + rng.gauss(0.0, sigma * (hi - lo))))
            cfg[key] = 10.0**perturbed
        elif isinstance(value, float):
            perturbed = float(value) + rng.gauss(0.0, sigma * (hi - lo))
            cfg[key] = float(max(lo, min(hi, perturbed)))
        elif isinstance(value, int):
            perturbed = round(float(value) + rng.gauss(0.0, sigma * (hi - lo)))
            cfg[key] = int(max(lo, min(hi, perturbed)))

    cfg["n_heads"] = snap_n_heads(int(cfg.get("d_model", 64)), int(cfg.get("n_heads", 4)))
    cfg["max_lr"] = max(float(cfg.get("max_lr", 4e-4)), float(cfg.get("lr", 2.5e-4)) * 1.05)
    if "batch_size" in cfg and rng.random() < 0.35:
        cfg["batch_size"] = int(rng.choice(_BATCH_SIZE_CHOICES))
    elif "batch_size" in cfg:
        cfg["batch_size"] = snap_batch_size(int(cfg["batch_size"]))
    return cfg


def run_primo_hpo(
    *,
    train_ds: Any,
    val_ds: Any,
    model_factory: Callable[[dict[str, Any]], Any],
    criterion_factory: Callable[[dict[str, Any]], Any],
    n_classes: int,
    n_channels: int,
    n_sectors: int,
    n_tickers: int,
    n_companies: int,
    device: Any,
    max_evaluations_total: int = 16,
    batch_size: int = 64,
    train_fraction: float = 0.50,
    val_fraction: float = 1.00,
    random_state: int = 42,
    root_directory: str | Path | None = None,
    overwrite_existing: bool = False,
    artifact_dir: str | Path | None = None,
    second_stage_top_k: int = 3,
    second_stage_epochs: int = 40,
) -> dict[str, Any]:
    """Run PriMO HPO through NePS when available, otherwise use fallback."""
    root = Path(root_directory or "/kaggle/working/credit_rating_artifacts/neps_primo")
    artifact_root = Path(artifact_dir) if artifact_dir is not None else root.parent
    if overwrite_existing and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    _TRIAL_ROWS.clear()
    configure_primo_context(
        train_ds=train_ds,
        val_ds=val_ds,
        model_factory=model_factory,
        criterion_factory=criterion_factory,
        n_classes=n_classes,
        n_channels=n_channels,
        n_sectors=n_sectors,
        n_tickers=n_tickers,
        n_companies=n_companies,
        device=device,
        batch_size=batch_size,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        random_state=random_state,
        artifact_dir=artifact_root,
    )

    if _NEPS_AVAILABLE:
        try:
            pipeline_space = build_pipeline_space()
            print("[NePS] Running PriMO HPO:")
            print(f"  Root dir:       {root}")
            print(f"  Total trials:   {max_evaluations_total}")
            print("  Fidelity range: epochs in [5, 20]")
            print("  Objectives:     [1-aligned_score (primary), val_loss (secondary)]")
            try:
                neps.run(  # type: ignore[union-attr]
                    evaluate_pipeline=evaluate_pipeline,
                    pipeline_space=pipeline_space,
                    root_directory=str(root),
                    max_evaluations_total=int(max_evaluations_total),
                    overwrite_working_directory=bool(overwrite_existing),
                    optimizer="PriMO",
                )
            except TypeError:
                neps.run(  # type: ignore[union-attr]
                    evaluate_pipeline=evaluate_pipeline,
                    pipeline_space=pipeline_space,
                    root_directory=str(root),
                    max_evaluations_total=int(max_evaluations_total),
                    overwrite_working_directory=bool(overwrite_existing),
                )
            _rerun_top_configs(root, top_k=int(second_stage_top_k), epochs=int(second_stage_epochs))
            best_config = _extract_best_config_from_neps(root)
        except Exception as exc:
            print(f"[NePS WARN] Falling back to custom PriMO after failure: {exc}")
            best_config = _run_fallback_primo(dict(PRIMO_EXPERT_PRIOR), int(max_evaluations_total), root)
            _rerun_top_configs(root, top_k=int(second_stage_top_k), epochs=int(second_stage_epochs))
            best_config = _extract_best_config_from_neps(root)
    else:
        best_config = _run_fallback_primo(dict(PRIMO_EXPERT_PRIOR), int(max_evaluations_total), root)
        _rerun_top_configs(root, top_k=int(second_stage_top_k), epochs=int(second_stage_epochs))
        best_config = _extract_best_config_from_neps(root)

    selected_path = artifact_root / "transformer_primo_selected_config.json"
    with selected_path.open("w", encoding="utf-8") as handle:
        json.dump(best_config, handle, indent=2)
    _write_top_configs(artifact_root, top_k=5)
    print(f"[PriMO-NePS] Config saved -> {selected_path}")
    return best_config


def load_primo_best_config(artifact_dir: str | Path) -> dict[str, Any]:
    """Load the selected PriMO config saved by Cell 24."""
    config_path = Path(artifact_dir) / "transformer_primo_selected_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
