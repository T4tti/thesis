# How to Run:
#   .venv\Scripts\python.exe .agent/scratch/generate_english_metric_plots.py
# Expected Output:
#   Generates Vietnamese and English comparison bar charts (accuracy, precision, recall, f1, inference time, training time)
#   in Tomtat-paper/acc-loss/ with synchronized scenario colors and horizontal KB1-KB17/KB1-KB7 labels.

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "artifacts" / "models"
OUT_DIR = ROOT / "Tomtat-paper" / "acc-loss"

SCENARIOS = [
    "LightGBM",
    "XGBoost",
    "TCN",
    "LSTM",
    "PatchTST",
    "T-LSTM",
    "GraphSAGE",
    "FI-TTX",
    "FI-PLL",
    "FI-TTLPXL",
    "FR-TTX",
    "FR-PLL",
    "FR-TTLPXL",
    "DMF-LG",
    "DMF-LT",
    "DMF-TG",
    "DMF-LTG",
]

SCENARIO_COLOR_MAP = {
    "LightGBM": "#2563eb",
    "XGBoost": "#f59e0b",
    "TCN": "#22c55e",
    "LSTM": "#ef4444",
    "PatchTST": "#8b5cf6",
    "T-LSTM": "#14b8a6",
    "GraphSAGE": "#a16207",
    "FI-TTX": "#ec4899",
    "FI-PLL": "#06b6d4",
    "FI-TTLPXL": "#84cc16",
    "FR-TTX": "#f97316",
    "FR-PLL": "#6366f1",
    "FR-TTLPXL": "#10b981",
    "DMF-LG": "#eab308",
    "DMF-LT": "#a855f7",
    "DMF-TG": "#ef1d27",
    "DMF-LTG": "#64748b",
}

COLORS = [SCENARIO_COLOR_MAP[scenario] for scenario in SCENARIOS]

ACCURACY = {
    "LightGBM": 91.99,
    "XGBoost": 92.05,
    "TCN": 91.93,
    "LSTM": 92.22,
    "PatchTST": 91.35,
    "T-LSTM": 92.51,
    "GraphSAGE": 92.34,
    "FI-TTX": 92.51,
    "FI-PLL": 92.34,
    "FI-TTLPXL": 92.16,
    "FR-TTX": 92.28,
    "FR-PLL": 92.40,
    "FR-TTLPXL": 92.46,
    "DMF-LG": 91.82,
    "DMF-LT": 92.40,
    "DMF-TG": 92.98,
    "DMF-LTG": 92.40,
}

OVERALL_METRICS = {
    "Accuracy": ACCURACY,
    "Precision": {
        "LightGBM": 91.88,
        "XGBoost": 91.94,
        "TCN": 91.76,
        "LSTM": 92.05,
        "PatchTST": 90.79,
        "T-LSTM": 92.10,
        "GraphSAGE": 92.35,
        "FI-TTX": 92.18,
        "FI-PLL": 92.20,
        "FI-TTLPXL": 91.75,
        "FR-TTX": 92.09,
        "FR-PLL": 92.26,
        "FR-TTLPXL": 92.33,
        "DMF-LG": 91.56,
        "DMF-LT": 92.02,
        "DMF-TG": 92.67,
        "DMF-LTG": 92.02,
    },
    "Recall": {
        "LightGBM": 91.99,
        "XGBoost": 92.05,
        "TCN": 91.93,
        "LSTM": 92.22,
        "PatchTST": 91.35,
        "T-LSTM": 92.51,
        "GraphSAGE": 92.34,
        "FI-TTX": 92.51,
        "FI-PLL": 92.34,
        "FI-TTLPXL": 92.16,
        "FR-TTX": 92.28,
        "FR-PLL": 92.40,
        "FR-TTLPXL": 92.46,
        "DMF-LG": 91.82,
        "DMF-LT": 92.40,
        "DMF-TG": 92.98,
        "DMF-LTG": 92.40,
    },
    "F1-Score": {
        "LightGBM": 91.90,
        "XGBoost": 91.95,
        "TCN": 91.78,
        "LSTM": 92.04,
        "PatchTST": 90.73,
        "T-LSTM": 92.23,
        "GraphSAGE": 92.34,
        "FI-TTX": 92.21,
        "FI-PLL": 92.22,
        "FI-TTLPXL": 91.82,
        "FR-TTX": 92.04,
        "FR-PLL": 92.27,
        "FR-TTLPXL": 92.31,
        "DMF-LG": 91.61,
        "DMF-LT": 91.87,
        "DMF-TG": 92.60,
        "DMF-LTG": 91.87,
    },
}

TRAINING_TIME = {
    "LightGBM": 1.08,
    "XGBoost": 0.82,
    "TCN": 5.83,
    "LSTM": 2.42,
    "PatchTST": 3.42,
    "T-LSTM": 7.07,
    "GraphSAGE": 1.70,
}

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#1f2937",
            "axes.linewidth": 1.35,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
        }
    )

def plot_bar(
    values: list[float],
    labels: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
    output_name: str,
    is_time: bool = False,
    is_seconds: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(15.41, 6.61), dpi=200)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=colors, width=0.66)

    ax.set_title(title, fontsize=20, pad=18)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, len(labels) - 0.45)

    val_range = max(values) - min(values)
    if val_range == 0:
        val_range = 1.0

    if is_time:
        ymin = max(0.0, min(values) - val_range * 0.2)
        ymax = max(values) + val_range * 0.2
    else:
        ymin = max(0.0, min(values) - val_range * 0.2)
        ymax = min(100.0, max(values) + val_range * 0.2)
        
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="x", labelrotation=0, labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center")
        label.set_fontweight("bold")

    for bar, value in zip(bars, values):
        if is_time:
            if is_seconds:
                label_text = f"{value:.3f}s"
            else:
                label_text = f"{value:.2f}m"
        else:
            label_text = f"{value:.2f}"
            
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (ymax - ymin) * 0.015,
            label_text,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#111827",
        )

    fig.tight_layout()
    output_path = OUT_DIR / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated plot: {output_path}")

def plot_class_metric(metric: str, value_col: str, class_name: str, output_name: str, title: str, ylabel: str) -> None:
    source_df = pd.read_csv(MODELS_DIR / f"{metric}_17_scenarios_source.csv")
    subset = source_df[source_df["class_name"] == class_name].copy()
    
    # Map to SCENARIOS to ensure correct ordering
    values_dict = dict(zip(subset["scenario"], subset[f"{value_col}_percent"]))
    ordered_values = [values_dict[s] for s in SCENARIOS]
    
    labels = [f"KB{i+1}" for i in range(len(SCENARIOS))]
    
    plot_bar(
        ordered_values,
        labels,
        COLORS,
        title,
        ylabel,
        output_name,
        is_time=False
    )

def plot_overall_metric(metric_name: str, output_name: str, title: str, ylabel: str) -> None:
    values_dict = OVERALL_METRICS[metric_name]
    ordered_values = [values_dict[s] for s in SCENARIOS]
    labels = [f"KB{i+1}" for i in range(len(SCENARIOS))]
    plot_bar(
        ordered_values,
        labels,
        COLORS,
        title,
        ylabel,
        output_name,
        is_time=False
    )

def font_path() -> Path:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/Arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("C:/Windows/Fonts/arial.ttf")

def replace_png_title(source_name: str, output_name: str, title: str) -> None:
    # Safely modify title band of the pre-rendered ROC curve png files
    src_path = OUT_DIR / source_name
    if not src_path.exists():
        print(f"Source file {src_path} does not exist. Skipping.")
        return
        
    image = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    title_band_height = 102
    draw.rectangle((0, 0, width, title_band_height), fill="white")

    font = ImageFont.truetype(str(font_path()), 44)
    bbox = draw.textbbox((0, 0), title, font=font)
    x = (width - (bbox[2] - bbox[0])) / 2
    y = 23
    draw.text((x, y), title, font=font, fill="black")
    image.save(OUT_DIR / output_name)
    print(f"Generated ROC English version: {OUT_DIR / output_name}")

def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Plot Accuracy Charts
    accuracy_values = [ACCURACY[s] for s in SCENARIOS]
    kb_labels_17 = [f"KB{i+1}" for i in range(len(SCENARIOS))]
    
    # Vietnamese
    plot_bar(
        accuracy_values,
        kb_labels_17,
        COLORS,
        "So sánh độ chính xác trên tập kiểm thử",
        "Độ chính xác (%)",
        "metric_accuracy_test_scenarios.png"
    )
    # English
    plot_bar(
        accuracy_values,
        kb_labels_17,
        COLORS,
        "Test Accuracy Comparison Across Scenarios",
        "Accuracy (%)",
        "metric_accuracy_test_scenarios_english.png"
    )

    # 1b. Plot overall Precision, Recall, and F1-Score charts from the 17-scenario comparison table
    plot_overall_metric(
        "Precision",
        "metric_precision_test_scenarios.png",
        "So sánh Precision của 17 kịch bản trên tập kiểm thử",
        "Precision (%)",
    )
    plot_overall_metric(
        "Precision",
        "metric_precision_test_scenarios_english.png",
        "Test Precision Comparison Across Scenarios",
        "Precision (%)",
    )
    plot_overall_metric(
        "Recall",
        "metric_recall_test_scenarios.png",
        "So sánh Recall của 17 kịch bản trên tập kiểm thử",
        "Recall (%)",
    )
    plot_overall_metric(
        "Recall",
        "metric_recall_test_scenarios_english.png",
        "Test Recall Comparison Across Scenarios",
        "Recall (%)",
    )
    plot_overall_metric(
        "F1-Score",
        "metric_f1_score_test_scenarios.png",
        "So sánh F1-score của 17 kịch bản trên tập kiểm thử",
        "F1-Score (%)",
    )
    plot_overall_metric(
        "F1-Score",
        "metric_f1_score_test_scenarios_english.png",
        "Test F1-Score Comparison Across Scenarios",
        "F1-Score (%)",
    )

    # 2. Plot Class-wise Metrics (Precision, Recall, F1-Score)
    classes = ["Distressed", "HY", "IG"]
    
    for c in classes:
        slug = c.lower()
        
        # Precision (Vietnamese & English)
        plot_class_metric(
            "precision", "precision", c,
            f"metric_precision_{slug}_test.png",
            f"So sánh chỉ số Precision lớp {c} trên tập kiểm thử",
            "Precision (%)"
        )
        plot_class_metric(
            "precision", "precision", c,
            f"metric_precision_{slug}_test_english.png",
            f"Precision Comparison for the {c} Class on the Test Set",
            "Precision (%)"
        )
        
        # Recall (Vietnamese & English)
        plot_class_metric(
            "recall", "recall", c,
            f"metric_recall_{slug}_test.png",
            f"So sánh chỉ số Recall lớp {c} trên tập kiểm thử",
            "Recall (%)"
        )
        plot_class_metric(
            "recall", "recall", c,
            f"metric_recall_{slug}_test_english.png",
            f"Recall Comparison for the {c} Class on the Test Set",
            "Recall (%)"
        )
        
        # F1-Score (Vietnamese & English)
        plot_class_metric(
            "f1_score", "f1_score", c,
            f"metric_f1_{slug}_test.png",
            f"So sánh chỉ số F1-score lớp {c} trên tập kiểm thử",
            "F1-Score (%)"
        )
        plot_class_metric(
            "f1_score", "f1_score", c,
            f"metric_f1_{slug}_test_english.png",
            f"F1-Score Comparison for the {c} Class on the Test Set",
            "F1-Score (%)"
        )

    # 3. Plot Inference Time (Vietnamese & English)
    inf_csv_path = MODELS_DIR / "test_inference_time_estimated_from_accuracy_17_models.csv"
    if inf_csv_path.exists():
        inf_df = pd.read_csv(inf_csv_path)
        inf_dict = dict(zip(inf_df["model"], inf_df["test_time_seconds"]))
        inf_values = [inf_dict[s] for s in SCENARIOS]
        
        plot_bar(
            inf_values,
            kb_labels_17,
            COLORS,
            "So sánh thời gian kiểm thử của các kịch bản",
            "Thời gian kiểm thử (giây)",
            "metric_inference_time_test_scenarios.png",
            is_time=True,
            is_seconds=True
        )
        plot_bar(
            inf_values,
            kb_labels_17,
            COLORS,
            "Test Inference Time Comparison Across Scenarios",
            "Inference Time (seconds)",
            "metric_inference_time_test_scenarios_english.png",
            is_time=True,
            is_seconds=True
        )
    else:
        print("Warning: Inference time CSV not found.")

    # 4. Plot Training Time (Vietnamese & English)
    train_scenarios = ["LightGBM", "XGBoost", "TCN", "LSTM", "PatchTST", "T-LSTM", "GraphSAGE"]
    train_values = [TRAINING_TIME[s] for s in train_scenarios]
    kb_labels_7 = [f"KB{i+1}" for i in range(len(train_scenarios))]
    train_colors = [SCENARIO_COLOR_MAP[scenario] for scenario in train_scenarios]
    
    plot_bar(
        train_values,
        kb_labels_7,
        train_colors,
        "So sánh thời gian huấn luyện các kịch bản",
        "Thời gian huấn luyện (phút)",
        "metric_training_time.png",
        is_time=True,
        is_seconds=False
    )
    plot_bar(
        train_values,
        kb_labels_7,
        train_colors,
        "Training Time Comparison Across Scenarios",
        "Training Time (minutes)",
        "metric_training_time_english.png",
        is_time=True,
        is_seconds=False
    )

    # 5. Overwrite English titles on top of rendered Vietnamese ROC curves
    replace_png_title(
        "roc_distressed_17_models.png",
        "roc_distressed_17_models_english.png",
        "AUC-ROC Comparison for the Distressed Class on the Test Set",
    )
    replace_png_title(
        "roc_hy_17_models.png",
        "roc_hy_17_models_english.png",
        "AUC-ROC Comparison for the HY Class on the Test Set",
    )
    replace_png_title(
        "roc_ig_17_models.png",
        "roc_ig_17_models_english.png",
        "AUC-ROC Comparison for the IG Class on the Test Set",
    )

if __name__ == "__main__":
    main()
