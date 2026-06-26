from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT_DIR / "result.xlsx"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "models"
COMPARISON_CSV_PATH = OUTPUT_DIR / "model_comparison_source.csv"
AUDIT_CSV_PATH = OUTPUT_DIR / "model_comparison_source_audit.csv"

SCENARIO_COL = "Kịch bản"
NAME_COL = "Tên kịch bản"
METHOD_COL = "Phương pháp"
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]


def read_csv_utf8(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def notebook_output_text(name: str) -> str:
    path = ROOT_DIR / "notebooks" / name
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy notebook: {path}")
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                text = output["text"]
                chunks.append("".join(text) if isinstance(text, list) else str(text))
            data = output.get("data", {})
            for key in ("text/plain", "text/html"):
                if key in data:
                    text = data[key]
                    chunks.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(chunks)


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [SCENARIO_COL, NAME_COL, METHOD_COL]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    return df


def base_row(kb: str, name: str, method: str, source: str) -> dict[str, object]:
    return {SCENARIO_COL: kb, NAME_COL: name, METHOD_COL: method, "Source": source}


def metric_row(
    kb: str,
    name: str,
    method: str,
    *,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    auc: float,
    source: str,
) -> dict[str, object]:
    row = base_row(kb, name, method, source)
    row.update(
        {
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
            "AUC": float(auc),
        }
    )
    return row


def report_csv_metrics(path: Path, auc_fallback: float | None = None) -> dict[str, float]:
    df = read_csv_utf8(path)
    lower_cols = {str(c).lower(): c for c in df.columns}

    if {"accuracy", "precision_weighted", "recall_weighted", "f1_weighted"}.issubset(lower_cols):
        row = df.iloc[0]
        auc_col = lower_cols.get("auc")
        auc = float(row[auc_col]) if auc_col is not None else auc_fallback
        if auc is None:
            raise ValueError(f"Không có AUC trong {path} và không có fallback")
        return {
            "accuracy": float(row[lower_cols["accuracy"]]),
            "precision": float(row[lower_cols["precision_weighted"]]),
            "recall": float(row[lower_cols["recall_weighted"]]),
            "f1": float(row[lower_cols["f1_weighted"]]),
            "auc": float(auc),
        }

    label_col = df.columns[0]
    labels = df[label_col].astype(str)
    weighted = df.loc[labels.eq("weighted avg")]
    accuracy = df.loc[labels.eq("accuracy")]
    if weighted.empty or accuracy.empty:
        raise ValueError(f"Không nhận dạng được format classification_report: {path}")
    auc = auc_fallback
    if auc is None:
        raise ValueError(f"Không có AUC trong {path} và không có fallback")
    weighted_row = weighted.iloc[0]
    accuracy_row = accuracy.iloc[0]
    return {
        "accuracy": float(accuracy_row["f1-score"]),
        "precision": float(weighted_row["precision"]),
        "recall": float(weighted_row["recall"]),
        "f1": float(weighted_row["f1-score"]),
        "auc": float(auc),
    }


def test_auc_from_metrics(path: Path, split_col: str = "Split", split_val: str = "test") -> float:
    df = read_csv_utf8(path)
    selected = df.loc[df[split_col].astype(str).str.lower().str.contains(split_val.lower())]
    if selected.empty:
        raise ValueError(f"Không tìm thấy split {split_val} trong {path}")
    return float(selected.iloc[0]["AUC"])


def row_from_report_csv(
    kb: str,
    name: str,
    method: str,
    report_path: Path,
    *,
    auc_fallback: float | None,
    source: str,
) -> dict[str, object]:
    metrics = report_csv_metrics(report_path, auc_fallback=auc_fallback)
    return metric_row(
        kb,
        name,
        method,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        auc=metrics["auc"],
        source=source,
    )


def row_from_metric_csv(kb: str, name: str, method: str, path: Path, source: str) -> dict[str, object]:
    row = read_csv_utf8(path).iloc[0]
    return metric_row(
        kb,
        name,
        method,
        accuracy=float(row["accuracy"]),
        precision=float(row["precision_weighted"]),
        recall=float(row["recall_weighted"]),
        f1=float(row["f1_weighted"]),
        auc=float(row["auc"]),
        source=source,
    )


def parse_test_summary_from_notebook(
    kb: str,
    name: str,
    method: str,
    notebook_name: str,
    source: str,
) -> dict[str, object]:
    text = notebook_output_text(notebook_name)
    
    report_pattern = re.compile(
        r"Classification Report[^\n]*:\s*\n\s*precision\s+recall\s+f1-score\s+support\s+"
        r".*?accuracy\s+(?P<accuracy>\d+\.\d+)\s+\d+\s+"
        r"macro avg\s+(?P<macro_p>\d+\.\d+)\s+(?P<macro_r>\d+\.\d+)\s+(?P<macro_f>\d+\.\d+)\s+\d+\s+"
        r"weighted avg\s+(?P<precision>\d+\.\d+)\s+(?P<recall>\d+\.\d+)\s+(?P<f1>\d+\.\d+)\s+\d+",
        re.S,
    )
    report_matches = list(report_pattern.finditer(text))
    if not report_matches:
        raise ValueError(f"Không tìm thấy classification report trong {notebook_name}")
    report = report_matches[-1]
    
    lines = [l.strip() for l in text.split("\n") if l.strip().startswith("Test")]
    if not lines:
        raise ValueError(f"Không tìm thấy dòng Test kết quả trong {notebook_name}")
    
    test_line = None
    for line in reversed(lines):
        parts = line.split()
        if len(parts) >= 15:
            try:
                float(parts[8])  # Accuracy
                float(parts[14]) # AUC
                test_line = parts
                break
            except (ValueError, IndexError):
                continue
                
    if test_line is None:
        raise ValueError(f"Không tìm thấy dòng Test kết quả có đủ cột trong {notebook_name}")
        
    accuracy = float(test_line[8])
    auc = float(test_line[14])
    
    return metric_row(
        kb,
        name,
        method,
        accuracy=accuracy,
        precision=float(report.group("precision")),
        recall=float(report.group("recall")),
        f1=float(report.group("f1")),
        auc=auc,
        source=source,
    )


def parse_gat_from_notebook() -> dict[str, object]:
    text = notebook_output_text("gat-baseline.ipynb")
    start = text.find("Test_Class0Calibrated")
    if start < 0:
        raise ValueError("Không tìm thấy dòng Test_Class0Calibrated trong gat-baseline.ipynb")
    block = text[start : start + 1800]
    head = re.search(
        r"Test_Class0Calibrated\s+\d+\.\d+\s+(?P<accuracy>\d+\.\d+)\s+(?P<precision>\d+\.\d+)",
        block,
    )
    middle = re.search(
        r"\n3\s+(?P<recall>\d+\.\d+)\s+(?P<macro>\d+\.\d+)\s+(?P<f1>\d+\.\d+)",
        block,
    )
    tail = re.search(
        r"\n3\s+\d+\.\d+\s+\d+\.\d+\s+\d+\s+(?P<auc>\d+\.\d+)\s+(?P<qwk>\d+\.\d+)",
        block,
    )
    if not (head and middle and tail):
        raise ValueError("Không đọc được metric Test_Class0Calibrated trong gat-baseline.ipynb")
    return metric_row(
        "KB13",
        "GAT",
        "",
        accuracy=float(head.group("accuracy")),
        precision=float(head.group("precision")),
        recall=float(middle.group("recall")),
        f1=float(middle.group("f1")),
        auc=float(tail.group("auc")),
        source="notebooks/gat-baseline.ipynb output: Test_Class0Calibrated + classification report",
    )


def parse_dmf_from_notebook() -> dict[str, object]:
    text = notebook_output_text("dynamic-model-fusion.ipynb")
    start = text.find("DMF/DCS proposed (constant)")
    if start < 0:
        raise ValueError("Không tìm thấy dòng DMF/DCS proposed trong dynamic-model-fusion.ipynb")
    block = text[start : start + 1400]
    head = re.search(
        r"DMF/DCS proposed \(constant\)\s+(?P<accuracy>\d+\.\d+)\s+"
        r"(?P<precision>\d+\.\d+)\s+(?P<recall>\d+\.\d+)",
        block,
    )
    tail = re.search(
        r"\n4\s+(?P<macro>\d+\.\d+)\s+(?P<f1>\d+\.\d+)\s+(?P<qwk>\d+\.\d+)\s+"
        r"(?P<mae>\d+\.\d+)\s+(?P<auc>\d+\.\d+)",
        block,
    )
    if not (head and tail):
        raise ValueError("Không đọc được metric DMF/DCS proposed trong dynamic-model-fusion.ipynb")
    return metric_row(
        "KB14",
        "DMF",
        "Dynamic Model Fusion (DMF)",
        accuracy=float(head.group("accuracy")),
        precision=float(head.group("precision")),
        recall=float(head.group("recall")),
        f1=float(tail.group("f1")),
        auc=float(tail.group("auc")),
        source="notebooks/dynamic-model-fusion.ipynb output: DMF/DCS proposed + classification report",
    )


def build_rows() -> list[dict[str, object]]:
    return [
        row_from_metric_csv(
            "KB1",
            "T-BiLSTM",
            "",
            ROOT_DIR / "artifacts" / "TLSTM" / "credit_rating_artifacts" / "transformer_metrics.csv",
            "artifacts/TLSTM/credit_rating_artifacts/transformer_metrics.csv; notebook classification report is rounded",
        ),
        row_from_report_csv(
            "KB2",
            "TCN",
            "",
            ROOT_DIR / "artifacts" / "TCN_Baseline" / "credit_rating_artifacts" / "tcn_test_classification_report.csv",
            auc_fallback=None,
            source="artifacts/TCN_Baseline/credit_rating_artifacts/tcn_test_classification_report.csv",
        ),
        row_from_report_csv(
            "KB3",
            "LSTM",
            "",
            ROOT_DIR / "artifacts" / "LSTM" / "credit_rating_artifacts" / "lstm_test_classification_report.csv",
            auc_fallback=test_auc_from_metrics(ROOT_DIR / "artifacts" / "LSTM" / "credit_rating_artifacts" / "lstm_metrics.csv"),
            source="artifacts/LSTM/credit_rating_artifacts/lstm_test_classification_report.csv + lstm_metrics.csv AUC",
        ),
        row_from_report_csv(
            "KB4",
            "PatchTST",
            "",
            ROOT_DIR / "artifacts" / "Patchtst" / "credit_rating_artifacts" / "patchtst_test_classification_report.csv",
            auc_fallback=test_auc_from_metrics(ROOT_DIR / "artifacts" / "Patchtst" / "credit_rating_artifacts" / "patchtst_metrics.csv"),
            source="artifacts/Patchtst/credit_rating_artifacts/patchtst_test_classification_report.csv + patchtst_metrics.csv AUC",
        ),
        row_from_report_csv(
            "KB5",
            "XGBoost",
            "",
            ROOT_DIR / "artifacts" / "XGBoost" / "credit_rating_artifacts" / "xgboost_test_classification_report.csv",
            auc_fallback=None,
            source="artifacts/XGBoost/credit_rating_artifacts/xgboost_test_classification_report.csv",
        ),
        row_from_report_csv(
            "KB6",
            "LightGBM",
            "",
            ROOT_DIR / "artifacts" / "LightGBM" / "credit_rating_artifacts" / "lightgbm_test_classification_report.csv",
            auc_fallback=None,
            source="artifacts/LightGBM/credit_rating_artifacts/lightgbm_test_classification_report.csv",
        ),
        parse_test_summary_from_notebook(
            "KB7",
            "FI-TTX",
            "Fuzzy Choquet Integral",
            "kb7-fi-ttx.ipynb",
            "notebooks/kb7-fi-ttx.ipynb output: Classification Report + completion table",
        ),
        parse_test_summary_from_notebook(
            "KB8",
            "FI-PLL",
            "Fuzzy Choquet Integral",
            "kb8-fi-pll.ipynb",
            "notebooks/kb8-fi-pll.ipynb output: Classification Report + completion table",
        ),
        parse_test_summary_from_notebook(
            "KB9",
            "FI-TTLPXL",
            "Fuzzy Choquet Integral",
            "kb9-fi-ttlpxl.ipynb",
            "notebooks/kb9-fi-ttlpxl.ipynb output: Classification Report fallback",
        ),
        parse_test_summary_from_notebook(
            "KB10",
            "FR-TTX",
            "Gompertz Fuzzy Ranking Ensemble",
            "kb10-fr-ttx.ipynb",
            "notebooks/kb10-fr-ttx.ipynb output: Classification Report + completion table",
        ),
        parse_test_summary_from_notebook(
            "KB11",
            "FR-PLL",
            "Gompertz Fuzzy Ranking Ensemble",
            "kb11-fr-pll.ipynb",
            "notebooks/kb11-fr-pll.ipynb output: Classification Report + completion table",
        ),
        parse_test_summary_from_notebook(
            "KB12",
            "FR-TTLPXL",
            "Gompertz Fuzzy Ranking Ensemble",
            "kb12-fr-ttlpxl.ipynb",
            "notebooks/kb12-fr-ttlpxl.ipynb output: Classification Report + completion table",
        ),
        row_from_report_csv(
            "KB13",
            "GAT",
            "",
            ROOT_DIR / "artifacts" / "GAT" / "credit_rating_artifacts" / "gat_test_classification_report.csv",
            auc_fallback=test_auc_from_metrics(
                ROOT_DIR / "artifacts" / "GAT" / "credit_rating_artifacts" / "gat_metrics.csv",
                split_val="test_class0calibrated"
            ),
            source="artifacts/GAT/credit_rating_artifacts/gat_test_classification_report.csv + gat_metrics.csv AUC",
        ),
        parse_dmf_from_notebook(),
    ]


def determine_group(method: str) -> str:
    lowered = str(method).lower()
    if "choquet" in lowered:
        return "FI Ensemble"
    if "gompertz" in lowered or "ranking" in lowered:
        return "FR Ensemble"
    if "dynamic" in lowered or "dmf" in lowered:
        return "DMF Ensemble"
    return "Baseline"


def plot_metric(df: pd.DataFrame, metric: str, output_name: str, title: str) -> Path:
    plot_df = df.copy()
    plot_df["Group"] = plot_df[METHOD_COL].map(determine_group)

    colors = {
        "Baseline": "#4A90E2",
        "FI Ensemble": "#1ABC9C",
        "FR Ensemble": "#E67E22",
        "DMF Ensemble": "#E74C3C",
    }
    labels = plot_df[SCENARIO_COL].tolist()
    values = plot_df[metric].astype(float).tolist()
    bar_colors = [colors[group] for group in plot_df["Group"]]

    y_min = max(0.0, min(values) - 0.015)
    y_max = min(1.0, max(values) + 0.015)
    if metric == "AUC":
        y_min = max(0.0, min(values) - 0.02)
        y_max = min(1.0, max(values) + 0.015)

    fig_width = max(12.5, len(labels) * 0.82)
    fig, ax = plt.subplots(figsize=(fig_width, 6.3))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="none", width=0.62, zorder=3)

    ax.set_title(f"So sánh chỉ số {title} của các mô hình (Test set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel(metric, fontsize=11, labelpad=10)
    ax.set_xlabel("Models", fontsize=11, labelpad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9.2, fontweight="semibold")

    legend_elements = [
        Patch(facecolor=colors["Baseline"], label="Mô hình cơ sở (Baseline)"),
        Patch(facecolor=colors["FI Ensemble"], label="Fuzzy Choquet (FI)"),
        Patch(facecolor=colors["FR Ensemble"], label="Gompertz Fuzzy Ranking (FR)"),
        Patch(facecolor=colors["DMF Ensemble"], label="Dynamic Model Fusion (DMF)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, facecolor="white", edgecolor="#DDDDDD")

    for bar in bars:
        height = float(bar.get_height())
        label_text = f"{height:.4f}" if metric == "AUC" else f"{height * 100:.2f}%"
        ax.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
            color="#333333",
        )

    plt.tight_layout()
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def upsert_rows_to_excel(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    if RESULT_PATH.exists():
        existing = pd.read_excel(RESULT_PATH, sheet_name="Sheet1")
    else:
        existing = pd.DataFrame(columns=[SCENARIO_COL, NAME_COL, METHOD_COL, *METRIC_COLUMNS])

    existing = normalize_text_columns(existing)
    rows_df = pd.DataFrame(list(rows))
    publish_df = rows_df[[SCENARIO_COL, NAME_COL, METHOD_COL, *METRIC_COLUMNS]].copy()

    for name in publish_df[NAME_COL].astype(str):
        existing = existing.loc[~existing[NAME_COL].astype(str).eq(name)].copy()
    combined = pd.concat([existing, publish_df], ignore_index=True)
    for col in METRIC_COLUMNS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    scenario_order = {f"KB{i}": i for i in range(1, 100)}
    combined["_scenario_order"] = combined[SCENARIO_COL].map(scenario_order).fillna(999)
    combined = combined.sort_values("_scenario_order").drop(columns="_scenario_order").reset_index(drop=True)
    combined.to_excel(RESULT_PATH, sheet_name="Sheet1", index=False)
    return combined


def main() -> None:
    rows = build_rows()
    comparison_df = upsert_rows_to_excel(rows)
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(AUDIT_CSV_PATH, index=False, encoding="utf-8-sig")

    outputs = [
        plot_metric(comparison_df, "Accuracy", "accuracy_comparison.png", "Accuracy"),
        plot_metric(comparison_df, "Precision", "precision_comparison.png", "Precision Weighted"),
        plot_metric(comparison_df, "Recall", "recall_comparison.png", "Recall Weighted"),
        plot_metric(comparison_df, "F1-Score", "f1_comparison.png", "F1-Score Weighted"),
        plot_metric(comparison_df, "AUC", "auc_comparison.png", "AUC-ROC"),
    ]

    print("Đã cập nhật biểu đồ từ classification report / notebook output:")
    print(COMPARISON_CSV_PATH)
    print(AUDIT_CSV_PATH)
    for path in outputs:
        print(f"Đã vẽ: {path}")


if __name__ == "__main__":
    main()
