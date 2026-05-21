# How to Run:
#   .venv\Scripts\python.exe scratch/generate_12_scenarios_plots.py
# Expected Output:
#   Đọc dữ liệu từ `result.xlsx`, làm sạch và vẽ 5 biểu đồ so sánh các chỉ số
#   Accuracy, Precision, Recall, F1-Score, AUC dưới thư mục `e:\thesis\artifacts\models`
#   (accuracy_comparison.png, precision_comparison.png, recall_comparison.png, f1_comparison.png, auc_comparison.png)
#   Các trục x (Models) sẽ hiển thị tên của các kịch bản.

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Cấu hình encoding utf-8 cho stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Đảm bảo kết quả vẽ biểu đồ chất lượng cao
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

# Khai báo các đường dẫn thư mục
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
EXCEL_PATH: Path = ROOT_DIR / "result.xlsx"
OUTPUT_DIR: Path = ROOT_DIR / "artifacts" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data_from_excel(file_path: Path) -> pd.DataFrame:
    """
    Đọc dữ liệu kết quả từ file Excel và thực hiện làm sạch dữ liệu.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Excel kết quả tại: {file_path}")
    
    # Đọc Sheet1
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    
    # Làm sạch tên cột
    df.columns = [str(c).strip() for c in df.columns]
    
    # Loại bỏ ký tự khoảng trắng đặc biệt (\xa0) và trim tên kịch bản
    if "Tên kịch bản" in df.columns:
        df["Tên kịch bản"] = df["Tên kịch bản"].apply(lambda x: str(x).replace("\xa0", " ").strip() if pd.notna(x) else "")
    else:
        raise KeyError("Không tìm thấy cột 'Tên kịch bản' trong file Excel.")
        
    if "Phương pháp" in df.columns:
        df["Phương pháp"] = df["Phương pháp"].apply(lambda x: str(x).replace("\xa0", " ").strip() if pd.notna(x) else "")
    
    # Xác định nhóm mô hình để tô màu
    df["Group"] = df.apply(determine_group, axis=1)
    
    return df

def determine_group(row: pd.Series) -> str:
    """
    Phân loại nhóm mô hình dựa trên phương pháp tổng hợp.
    """
    method = str(row.get("Phương pháp", "")).strip()
    if pd.isna(row.get("Phương pháp")) or method.lower() == "nan" or not method:
        return "Baseline"
    elif "choquet" in method.lower():
        return "FI Ensemble"
    elif "gompertz" in method.lower() or "ranking" in method.lower():
        return "FR Ensemble"
    return "Baseline"

def generate_plot(df: pd.DataFrame, metric_col: str, file_name: str, title_vn: str, y_min: float, y_max: float) -> None:
    """
    Vẽ biểu đồ cột so sánh một chỉ số đánh giá cho 12 kịch bản.
    """
    values: List[float] = df[metric_col].tolist()
    labels: List[str] = df["Tên kịch bản"].tolist()
    groups: List[str] = df["Group"].tolist()
    
    # Định nghĩa bảng màu cao cấp
    colors_map: Dict[str, str] = {
        "Baseline": "#4a90e2",      # Xanh lam dịu cho mô hình cơ sở
        "FI Ensemble": "#1abc9c",   # Xanh ngọc cho tích phân fuzzy Choquet
        "FR Ensemble": "#e67e22"    # Cam ấm cho Gompertz Fuzzy Ranking
    }
    bar_colors: List[str] = [colors_map[g] for g in groups]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Vẽ các cột biểu đồ
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='none', width=0.6, zorder=3)
    
    # Đặt tiêu đề và nhãn các trục
    ax.set_title(f"So sánh chỉ số {title_vn} của 12 kịch bản (Test set)", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel(metric_col, fontsize=11, labelpad=10)
    ax.set_xlabel("Models", fontsize=11, labelpad=10)
    
    # Tùy chỉnh lưới hiển thị và các đường biên (spines)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Đặt giới hạn trục y để làm nổi bật sự khác biệt nhỏ giữa các mô hình hiệu năng cao
    ax.set_ylim(y_min, y_max)
    
    # Xoay nhãn trục x để không bị đè lên nhau
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5, fontweight='semibold')
    
    # Thêm chú thích nhóm mô hình (Legend)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_map["Baseline"], label='Mô hình cơ sở (Baselines)'),
        Patch(facecolor=colors_map["FI Ensemble"], label='Tổng hợp Fuzzy Choquet (FI)'),
        Patch(facecolor=colors_map["FR Ensemble"], label='Tổng hợp Gompertz Fuzzy Ranking (FR)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='white', edgecolor='#dddddd')
    
    # Thêm số liệu cụ thể lên đỉnh mỗi cột
    for bar in bars:
        height = bar.get_height()
        # Đối với chỉ số AUC hiển thị dạng thập phân, còn lại hiển thị %
        if metric_col.upper() != "AUC":
            label_text = f"{height*100:.2f}%" if height <= 1.0 else f"{height:.2f}%"
        else:
            label_text = f"{height:.4f}"
            
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Độ lệch dọc 3 points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='semibold',
                    color='#333333')
                    
    plt.tight_layout()
    output_path: Path = OUTPUT_DIR / file_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated plot: {output_path}")

def main() -> None:
    # 1. Đọc dữ liệu từ file kết quả Excel
    df = load_data_from_excel(EXCEL_PATH)
    
    # 2. Vẽ 5 biểu đồ so sánh: Accuracy, Precision, Recall, F1-Score, AUC
    # Thiết lập giới hạn y_min và y_max thông minh để hiển thị rõ sự khác biệt của các mô hình
    generate_plot(df, "Accuracy", "accuracy_comparison.png", "Accuracy (Độ chính xác)", 0.88, 0.94)
    generate_plot(df, "Precision", "precision_comparison.png", "Precision (Weighted)", 0.88, 0.94)
    generate_plot(df, "Recall", "recall_comparison.png", "Recall (Weighted)", 0.88, 0.94)
    generate_plot(df, "F1-Score", "f1_comparison.png", "F1-Score (Weighted)", 0.88, 0.94)
    generate_plot(df, "AUC", "auc_comparison.png", "AUC-ROC", 0.93, 0.99)

if __name__ == "__main__":
    main()
