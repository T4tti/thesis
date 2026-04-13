"""Train HHGNN-Fuzzy model for corporate credit rating.

This pipeline upgrades the original HHGNN notebook into a reproducible script with:
- train-fit preprocessing
- fuzzy feature-graph construction
- class-balanced focal training with AMP
- OneCycleLR + early stopping
- full artifact and report export
- optional SHAP explainability

Run from repo root:
    python src/pipelines/train_hhgnn.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 16
    early_stop_min_delta: float = 1e-3

    hidden_channels: int = 96
    num_gat_layers: int = 3
    heads: int = 4
    dropout: float = 0.30
    fc_dropout: float = 0.25

    learning_rate: float = 4e-4
    max_learning_rate: float = 1.1e-3
    weight_decay: float = 8e-4

    focal_gamma: float = 1.8
    label_smoothing: float = 0.02
    cb_beta: float = 0.995

    fuzzy_alpha: float = 10.0
    fuzzy_beta: float = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HHGNN-Fuzzy training pipeline")

    parser.add_argument("--train-path", type=str, default="")
    parser.add_argument("--val-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--target-col", type=str, default="rating_detail")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=16)

    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--num-gat-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--fc-dropout", type=float, default=0.25)

    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--max-learning-rate", type=float, default=1.1e-3)
    parser.add_argument("--weight-decay", type=float, default=8e-4)

    parser.add_argument("--focal-gamma", type=float, default=1.8)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--cb-beta", type=float, default=0.995)

    parser.add_argument("--fuzzy-alpha", type=float, default=10.0)
    parser.add_argument("--fuzzy-beta", type=float, default=0.3)

    parser.add_argument("--output-model-dir", type=str, default="artifacts/models/hhgnn_fuzzy")
    parser.add_argument("--output-report-dir", type=str, default="data/reports")

    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-background", type=int, default=80)
    parser.add_argument("--shap-samples", type=int, default=150)
    parser.add_argument("--shap-nsamples", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_existing_path(path_hint: str, candidates: List[str]) -> Path:
    """Resolve explicit path first, otherwise first available candidate path."""
    paths: List[Path] = []

    if path_hint:
        p = Path(path_hint)
        if not p.is_absolute():
            p = ROOT_DIR / p
        paths.append(p)

    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = ROOT_DIR / p
        paths.append(p)

    for p in paths:
        if p.exists():
            return p

    tried = [str(x) for x in paths]
    raise FileNotFoundError(f"Khong tim thay file du lieu. Da thu: {tried}")


def build_train_sampler(labels: np.ndarray) -> tuple[torch.utils.data.WeightedRandomSampler | None, float]:
    """Create weighted sampler when class imbalance is meaningful."""
    class_freq = np.bincount(labels).astype(np.float64)
    non_zero = class_freq[class_freq > 0]
    if len(non_zero) == 0:
        return None, 1.0

    imbalance_ratio = float(non_zero.max() / non_zero.min())
    if imbalance_ratio < 2.0:
        return None, imbalance_ratio

    class_freq_safe = np.maximum(class_freq, 1.0)
    inv_freq = 1.0 / class_freq_safe
    sample_weights = inv_freq[labels]
    sample_weights = sample_weights / np.mean(sample_weights)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, imbalance_ratio


def evaluate_loader(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    n_classes: int,
    device: torch.device,
    amp_enabled: bool,
    metric_fn: Any,
) -> tuple[float, Dict[str, float], np.ndarray, np.ndarray, torch.Tensor]:
    """Run one full evaluation pass and return metrics + logits."""

    losses: List[float] = []
    y_true_all: List[np.ndarray] = []
    logits_all: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if amp_enabled and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = criterion(logits, batch.y)
            else:
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(logits, batch.y)

            losses.append(float(loss.item()))
            y_true_all.append(batch.y.detach().cpu().numpy())
            logits_all.append(logits.detach().cpu())

    y_true = np.concatenate(y_true_all, axis=0)
    logits_t = torch.cat(logits_all, dim=0)
    metrics = metric_fn(y_true=y_true, logits=logits_t, n_classes=n_classes)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_pred = torch.argmax(logits_t, dim=1).cpu().numpy()

    return mean_loss, metrics, y_true, y_pred, logits_t


def save_history_plot(history: Dict[str, List[float]], output_path: Path) -> None:
    """Save training curves (loss/metrics/lr)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    axes[0, 0].plot(history["train_loss"], label="Train", color="#1f77b4")
    axes[0, 0].plot(history["val_loss"], label="Val", color="#ff7f0e")
    axes[0, 0].set_title("Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(history["train_acc"], label="Train", color="#2ca02c")
    axes[0, 1].plot(history["val_acc"], label="Val", color="#d62728")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].plot(history["train_f1w"], label="Train", color="#9467bd")
    axes[0, 2].plot(history["val_f1w"], label="Val", color="#8c564b")
    axes[0, 2].set_title("F1 weighted")
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].legend()

    axes[1, 0].plot(history["train_auc"], label="Train", color="#17becf")
    axes[1, 0].plot(history["val_auc"], label="Val", color="#bcbd22")
    axes[1, 0].set_title("AUC (OvR weighted)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(history["train_mae"], label="Train", color="#e377c2")
    axes[1, 1].plot(history["val_mae"], label="Val", color="#7f7f7f")
    axes[1, 1].set_title("MAE")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].plot(history["lr"], color="#ff6600")
    axes[1, 2].set_title("Learning Rate")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_values: List[Any],
    output_path: Path,
) -> None:
    """Save confusion matrix figure."""
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = [str(label_values[i]) if i < len(label_values) else str(i) for i in labels]

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    if len(labels) <= 25:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
        )
    else:
        sns.heatmap(cm, cmap="Blues", ax=ax)

    ax.set_title("HHGNN Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_optional_shap(
    model: nn.Module,
    dataset_info: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_report_dir: Path,
    device: torch.device,
    alpha: float,
    beta: float,
    background_size: int,
    shap_samples: int,
    shap_nsamples: int,
) -> None:
    """Run optional SHAP KernelExplainer on tabular wrapper."""
    try:
        import shap  # type: ignore[import-not-found]
        from torch_geometric.data import Batch, Data  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"SHAP bo qua vi thieu thu vien: {exc}")
        return

    class HHGNNWrapper:
        def __init__(self, net: nn.Module):
            self.net = net
            self.net.eval()

        def predict(self, X_tabular: np.ndarray) -> np.ndarray:
            data_list = []
            d = X_tabular.shape[1]
            denom = max(d - 1, 1)

            for i in range(len(X_tabular)):
                feat = X_tabular[i]
                node_x = np.column_stack([
                    feat,
                    np.arange(d, dtype=np.float64) / denom,
                ])
                from src.models.hhgnn_fuzzy import build_fuzzy_feature_graph  # local import to avoid circular at module load

                edge_index, edge_weight = build_fuzzy_feature_graph(feat, alpha=alpha, beta=beta)
                data = Data(
                    x=torch.tensor(node_x, dtype=torch.float32),
                    edge_index=edge_index,
                    edge_attr=edge_weight.unsqueeze(1),
                )
                data_list.append(data)

            batch = Batch.from_data_list(data_list).to(device)
            with torch.no_grad():
                logits = self.net(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs

    X_all = dataset_info.X_all
    background = X_all[train_idx][: min(background_size, len(train_idx))]
    X_shap = X_all[test_idx][: min(shap_samples, len(test_idx))]

    if len(background) < 8 or len(X_shap) < 4:
        print("SHAP bo qua vi so mau qua it")
        return

    print(f"Dang chay SHAP: background={len(background)}, samples={len(X_shap)}")
    wrapper = HHGNNWrapper(model)
    explainer = shap.KernelExplainer(wrapper.predict, background)
    shap_values = explainer.shap_values(X_shap, nsamples=shap_nsamples)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=dataset_info.feature_names, show=False, plot_type="bar")
    plt.title("SHAP Feature Importance - HHGNN")
    plt.tight_layout()
    plt.savefig(output_report_dir / "hhgnn_fuzzy_shap_bar.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader  # type: ignore[import-not-found]
        from src.models.hhgnn_fuzzy import (
            DEFAULT_FINANCIAL_FEATURES,
            FuzzyFocalLoss,
            HHGNNFuzzyClassifier,
            build_feature_graph_dataset,
            classification_metrics,
            compute_effective_class_weights,
            compute_fuzzy_sample_weights,
            prepare_static_company_dataset,
            set_seed,
        )
    except Exception as exc:
        raise ModuleNotFoundError(
            "Thieu dependency cho HHGNN. Cai dat truoc khi train: "
            "pip install torch torch-geometric torch-scatter torch-sparse"
        ) from exc

    config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_channels=args.hidden_channels,
        num_gat_layers=args.num_gat_layers,
        heads=args.heads,
        dropout=args.dropout,
        fc_dropout=args.fc_dropout,
        learning_rate=args.learning_rate,
        max_learning_rate=args.max_learning_rate,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        cb_beta=args.cb_beta,
        fuzzy_alpha=args.fuzzy_alpha,
        fuzzy_beta=args.fuzzy_beta,
    )

    model_dir = Path(args.output_model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT_DIR / model_dir
    report_dir = Path(args.output_report_dir)
    if not report_dir.is_absolute():
        report_dir = ROOT_DIR / report_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()

    train_path = resolve_existing_path(
        args.train_path,
        [
            "data/processed/train_smote_augmented.csv",
            "data/processed/train_augmented_timegan_optimized.csv",
            "data/processed/train_augmented_timegan.csv",
            "archive/ctgan/splits/train_augmented_ctgan.csv",
            "data/processed/ctgan/splits/train_augmented_ctgan.csv",
        ],
    )
    val_path = resolve_existing_path(
        args.val_path,
        [
            "data/processed/val.csv",
            "data/processed/val_split.csv",
            "archive/ctgan/splits/val.csv",
            "data/processed/ctgan/splits/val.csv",
        ],
    )
    test_path = resolve_existing_path(
        args.test_path,
        [
            "data/processed/test.csv",
            "data/processed/test_split.csv",
            "archive/ctgan/splits/test.csv",
            "data/processed/ctgan/splits/test.csv",
        ],
    )

    print("=" * 90)
    print("HHGNN-FUZZY TRAINING PIPELINE")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Train file: {train_path}")
    print(f"Val file:   {val_path}")
    print(f"Test file:  {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    dataset_info = prepare_static_company_dataset(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        financial_features=DEFAULT_FINANCIAL_FEATURES,
        target_col=args.target_col,
    )

    n_classes = int(len(dataset_info.label_values))
    print(f"Static companies: {len(dataset_info.df_static)}")
    print(
        f"Train/Val/Test companies: "
        f"{len(dataset_info.train_idx)}/{len(dataset_info.val_idx)}/{len(dataset_info.test_idx)}"
    )
    print(f"N classes: {n_classes}")

    fuzzy_weights_train = compute_fuzzy_sample_weights(
        X=dataset_info.X_all[dataset_info.train_idx],
        y=dataset_info.y_all[dataset_info.train_idx],
        n_classes=n_classes,
    )

    fuzzy_weight_all = np.ones(len(dataset_info.X_all), dtype=np.float64)
    fuzzy_weight_all[dataset_info.train_idx] = fuzzy_weights_train

    train_data_list = build_feature_graph_dataset(
        X_all=dataset_info.X_all,
        y_all=dataset_info.y_all,
        indices=dataset_info.train_idx,
        sample_weights=fuzzy_weight_all,
        alpha=config.fuzzy_alpha,
        beta=config.fuzzy_beta,
    )
    val_data_list = build_feature_graph_dataset(
        X_all=dataset_info.X_all,
        y_all=dataset_info.y_all,
        indices=dataset_info.val_idx,
        sample_weights=None,
        alpha=config.fuzzy_alpha,
        beta=config.fuzzy_beta,
    )
    test_data_list = build_feature_graph_dataset(
        X_all=dataset_info.X_all,
        y_all=dataset_info.y_all,
        indices=dataset_info.test_idx,
        sample_weights=None,
        alpha=config.fuzzy_alpha,
        beta=config.fuzzy_beta,
    )

    train_labels = dataset_info.y_all[dataset_info.train_idx]
    train_sampler, imbalance_ratio = build_train_sampler(train_labels)

    train_loader = PyGDataLoader(
        train_data_list,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    val_loader = PyGDataLoader(val_data_list, batch_size=config.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_data_list, batch_size=config.batch_size, shuffle=False)

    print(f"Train imbalance ratio: {imbalance_ratio:.3f}")
    print(f"Weighted sampler enabled: {train_sampler is not None}")

    model = HHGNNFuzzyClassifier(
        in_channels=2,
        hidden_channels=config.hidden_channels,
        n_classes=n_classes,
        num_gat_layers=config.num_gat_layers,
        heads=config.heads,
        dropout=config.dropout,
        fc_dropout=config.fc_dropout,
    ).to(device)

    class_weights = compute_effective_class_weights(
        y_train=train_labels,
        n_classes=n_classes,
        beta=config.cb_beta,
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = FuzzyFocalLoss(
        gamma=config.focal_gamma,
        class_weight=class_weights_t,
        reduction="mean",
        label_smoothing=config.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_learning_rate,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=config.max_epochs,
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=max(config.max_learning_rate / max(config.learning_rate, 1e-8), 1.0),
        final_div_factor=100.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_model_path = model_dir / "hhgnn_fuzzy_best_model.pt"
    best_meta_path = model_dir / "hhgnn_fuzzy_best_model_meta.pt"

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1w": [],
        "val_f1w": [],
        "train_auc": [],
        "val_auc": [],
        "train_mae": [],
        "val_mae": [],
        "lr": [],
    }

    best_val_f1w = -np.inf
    best_epoch = -1
    patience_counter = 0
    best_state: Dict[str, torch.Tensor] | None = None

    print("\nBat dau train...")
    for epoch in range(config.max_epochs):
        model.train()
        train_losses: List[float] = []
        train_y: List[np.ndarray] = []
        train_logits: List[torch.Tensor] = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            fuzzy_w = None
            if hasattr(batch, "sample_weight"):
                fuzzy_w = batch.sample_weight.view(-1)

            if amp_enabled and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = criterion(logits, batch.y, fuzzy_weights=fuzzy_w)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(logits, batch.y, fuzzy_weights=fuzzy_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            train_losses.append(float(loss.item()))
            train_y.append(batch.y.detach().cpu().numpy())
            train_logits.append(logits.detach().cpu())

        tr_y = np.concatenate(train_y, axis=0)
        tr_logits = torch.cat(train_logits, dim=0)
        tr_metrics = classification_metrics(y_true=tr_y, logits=tr_logits, n_classes=n_classes)
        tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        val_loss, val_metrics, _, _, _ = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            n_classes=n_classes,
            device=device,
            amp_enabled=amp_enabled,
            metric_fn=classification_metrics,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1w"].append(tr_metrics["f1_weighted"])
        history["val_f1w"].append(val_metrics["f1_weighted"])
        history["train_auc"].append(tr_metrics["auc_ovr_weighted"])
        history["val_auc"].append(val_metrics["auc_ovr_weighted"])
        history["train_mae"].append(tr_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["lr"].append(current_lr)

        val_f1w = val_metrics["f1_weighted"]
        if val_f1w > best_val_f1w + config.early_stop_min_delta:
            best_val_f1w = val_f1w
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            torch.save(best_state, best_model_path)
            torch.save(
                {
                    "best_epoch": best_epoch,
                    "best_val_f1_weighted": float(best_val_f1w),
                    "n_classes": n_classes,
                    "feature_names": list(dataset_info.feature_names),
                    "label_values": list(dataset_info.label_values),
                    "target_col": args.target_col,
                    "config": asdict(config),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                },
                best_meta_path,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1:03d}/{config.max_epochs} | "
            f"TrLoss={tr_loss:.4f} VlLoss={val_loss:.4f} | "
            f"TrF1w={tr_metrics['f1_weighted']:.4f} VlF1w={val_metrics['f1_weighted']:.4f} | "
            f"VlAUC={val_metrics['auc_ovr_weighted']:.4f} | LR={current_lr:.7f}"
        )

        if patience_counter >= config.patience:
            print(f"Dung som tai epoch {epoch + 1} do khong cai thien val_f1_weighted")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print("\nDanh gia model tot nhat tren train/val/test...")

    train_eval_loss, train_eval_metrics, _, _, _ = evaluate_loader(
        model=model,
        loader=train_loader,
        criterion=criterion,
        n_classes=n_classes,
        device=device,
        amp_enabled=amp_enabled,
        metric_fn=classification_metrics,
    )
    val_eval_loss, val_eval_metrics, _, _, _ = evaluate_loader(
        model=model,
        loader=val_loader,
        criterion=criterion,
        n_classes=n_classes,
        device=device,
        amp_enabled=amp_enabled,
        metric_fn=classification_metrics,
    )
    test_eval_loss, test_eval_metrics, y_true_test, y_pred_test, logits_test = evaluate_loader(
        model=model,
        loader=test_loader,
        criterion=criterion,
        n_classes=n_classes,
        device=device,
        amp_enabled=amp_enabled,
        metric_fn=classification_metrics,
    )

    label_decoder = {
        i: str(dataset_info.label_values[i])
        for i in range(len(dataset_info.label_values))
    }

    report = str(classification_report(
        y_true_test,
        y_pred_test,
        labels=sorted(set(y_true_test.tolist()) | set(y_pred_test.tolist())),
        target_names=[label_decoder.get(i, str(i)) for i in sorted(set(y_true_test.tolist()) | set(y_pred_test.tolist()))],
        zero_division=0,
    ))

    metrics_rows = [
        {
            "split": "train",
            "loss": train_eval_loss,
            **train_eval_metrics,
        },
        {
            "split": "val",
            "loss": val_eval_loss,
            **val_eval_metrics,
        },
        {
            "split": "test",
            "loss": test_eval_loss,
            **test_eval_metrics,
            "best_epoch": best_epoch,
            "best_val_f1_weighted": float(best_val_f1w),
        },
    ]

    metrics_path = report_dir / "hhgnn_fuzzy_metrics.csv"
    history_path = report_dir / "hhgnn_fuzzy_history.csv"
    report_path = report_dir / "hhgnn_fuzzy_classification_report.txt"
    pred_path = report_dir / "hhgnn_fuzzy_predictions.csv"
    curves_path = report_dir / "hhgnn_fuzzy_metric_curves.png"
    cm_path = report_dir / "hhgnn_fuzzy_confusion_matrix.png"
    config_path = report_dir / "hhgnn_fuzzy_train_config.json"

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8")
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8")
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)

    test_rows = dataset_info.df_static.iloc[dataset_info.test_idx].copy()
    probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
    confidence = probs_test.max(axis=1)

    pred_df = pd.DataFrame(
        {
            "ticker": test_rows["ticker"].astype("string").to_numpy(),
            "sector": test_rows["sector"].astype("string").to_numpy(),
            "y_true_idx": y_true_test,
            "y_pred_idx": y_pred_test,
            "y_true_label": [label_decoder.get(int(x), str(x)) for x in y_true_test],
            "y_pred_label": [label_decoder.get(int(x), str(x)) for x in y_pred_test],
            "confidence": confidence,
        }
    )
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    save_history_plot(history=history, output_path=curves_path)
    save_confusion_plot(y_true=y_true_test, y_pred=y_pred_test, label_values=dataset_info.label_values, output_path=cm_path)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print("\nKet qua test:")
    print(
        f"  Accuracy={test_eval_metrics['accuracy']:.4f} | "
        f"F1w={test_eval_metrics['f1_weighted']:.4f} | "
        f"AUC={test_eval_metrics['auc_ovr_weighted']:.4f} | "
        f"MAE={test_eval_metrics['mae']:.4f}"
    )
    print("Artifacts da luu:")
    print(f"  Model: {best_model_path}")
    print(f"  Meta:  {best_meta_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Report:  {report_path}")
    print(f"  Preds:   {pred_path}")

    if args.run_shap:
        run_optional_shap(
            model=model,
            dataset_info=dataset_info,
            train_idx=dataset_info.train_idx,
            test_idx=dataset_info.test_idx,
            output_report_dir=report_dir,
            device=device,
            alpha=config.fuzzy_alpha,
            beta=config.fuzzy_beta,
            background_size=args.shap_background,
            shap_samples=args.shap_samples,
            shap_nsamples=args.shap_nsamples,
        )


if __name__ == "__main__":
    main()
