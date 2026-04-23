from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "reports" / "model_metric_comparison"

MODEL_FILES = {
    "LSTM": ROOT_DIR / "artifacts" / "LSTM" / "LSTM" / "lstm_metrics.csv",
    "PatchTST": ROOT_DIR / "artifacts" / "Patchtst" / "Patchtst" / "patchtst_metrics.csv",
    "TCN": ROOT_DIR / "artifacts" / "TCN_Baseline" / "TCN" / "tcn_metrics.csv",
    "TLSTM": ROOT_DIR / "artifacts" / "TLSTM" / "TLSTM" / "transformer_metrics.csv",
}

METRIC_ALIASES = {
    "accuracy": ["accuracy"],
    "f1_macro": ["f1_macro", "macro_f1"],
    "qwk": ["qwk"],
    "auc": ["auc"],
}


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str:
    col_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        key = alias.lower().strip()
        if key in col_map:
            return col_map[key]
    raise KeyError(f"Khong tim thay cot cho aliases={aliases}. Cac cot hien co: {list(df.columns)}")


def _pick_test_row(df: pd.DataFrame) -> pd.Series:
    col_map = {c.lower().strip(): c for c in df.columns}
    if "split" not in col_map:
        return df.iloc[0]

    split_col = col_map["split"]
    test_rows = df[df[split_col].astype(str).str.lower() == "test"]
    if test_rows.empty:
        raise ValueError("File co cot Split nhung khong co dong Test.")
    return test_rows.iloc[0]


def load_model_metrics(file_path: Path) -> dict[str, float]:
    if not file_path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {file_path}")

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"File rong: {file_path}")

    row = _pick_test_row(df)

    result: dict[str, float] = {}
    for metric_key, aliases in METRIC_ALIASES.items():
        column = _find_column(df, aliases)
        result[metric_key] = float(row[column])

    return result


def plot_metric_bars(comparison_df: pd.DataFrame, metric_key: str, metric_title: str) -> Path:
    values = comparison_df[metric_key].tolist()
    models = comparison_df["model"].tolist()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values)
    plt.title(f"So sanh {metric_title} tren tap Test")
    plt.ylabel(metric_title)
    plt.ylim(0.0, min(1.0, max(values) + 0.08))
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{metric_key}_comparison.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, file_path in MODEL_FILES.items():
        metrics = load_model_metrics(file_path)
        rows.append({"model": model_name, **metrics})

    comparison_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "model_test_metrics_comparison.csv"
    comparison_df.to_csv(summary_path, index=False, encoding="utf-8")

    metric_titles = {
        "accuracy": "Accuracy",
        "f1_macro": "F1 Macro",
        "qwk": "QWK",
        "auc": "AUC",
    }

    print("Da tong hop metric tren tap Test:")
    print(comparison_df.to_string(index=False))
    print(f"\nDa luu bang tong hop: {summary_path}")

    for metric_key, metric_title in metric_titles.items():
        out_file = plot_metric_bars(comparison_df, metric_key, metric_title)
        print(f"Da luu bieu do: {out_file}")


if __name__ == "__main__":
    main()
