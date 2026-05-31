import pandas as pd
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
EXCEL_PATH = ROOT_DIR / "result.xlsx"
PLOTTING_SCRIPT = ROOT_DIR / "scratch" / "generate_12_scenarios_plots.py"

# Compiled metrics from latest notebook runs for KB1-12
updated_metrics = {
    "KB1": { # T-BiLSTM
        "Accuracy": 0.925131,
        "Precision": 0.929439,
        "Recall": 0.925131,
        "F1-Score": 0.926714,
        "AUC": 0.980335
    },
    "KB2": { # TCN
        "Accuracy": 0.923970,
        "Precision": 0.924500,
        "Recall": 0.923970,
        "F1-Score": 0.924201,
        "AUC": 0.950575
    },
    "KB3": { # LSTM
        "Accuracy": 0.925131,
        "Precision": 0.925793,
        "Recall": 0.925131,
        "F1-Score": 0.925439,
        "AUC": 0.947931
    },
    "KB4": { # PatchTST
        "Accuracy": 0.923970,
        "Precision": 0.924400,
        "Recall": 0.923970,
        "F1-Score": 0.924165,
        "AUC": 0.952839
    },
    "KB5": { # XGBoost
        "Accuracy": 0.919907,
        "Precision": 0.918411,
        "Recall": 0.919907,
        "F1-Score": 0.918528,
        "AUC": 0.962333
    },
    "KB6": { # LightGBM
        "Accuracy": 0.918746,
        "Precision": 0.917803,
        "Recall": 0.918746,
        "F1-Score": 0.918020,
        "AUC": 0.963067
    },
    "KB7": { # FI-TTX
        "Accuracy": 0.922809,
        "Precision": 0.921866,
        "Recall": 0.922809,
        "F1-Score": 0.922087,
        "AUC": 0.960217
    },
    "KB8": { # FI-PLL
        "Accuracy": 0.923970,
        "Precision": 0.923573,
        "Recall": 0.923970,
        "F1-Score": 0.923660,
        "AUC": 0.965804
    },
    "KB9": { # FI-TTLPXL
        "Accuracy": 0.921648,
        "Precision": 0.920139,
        "Recall": 0.921648,
        "F1-Score": 0.920412,
        "AUC": 0.961228
    },
    "KB10": { # FR-TTX
        "Accuracy": 0.919327,
        "Precision": 0.916550,
        "Recall": 0.919327,
        "F1-Score": 0.917088,
        "AUC": 0.959230
    },
    "KB11": { # FR-PLL
        "Accuracy": 0.923970,
        "Precision": 0.923573,
        "Recall": 0.923970,
        "F1-Score": 0.923660,
        "AUC": 0.965804
    },
    "KB12": { # FR-TTLPXL
        "Accuracy": 0.922229,
        "Precision": 0.921115,
        "Recall": 0.922229,
        "F1-Score": 0.921379,
        "AUC": 0.961548
    }
}

def load_gat_metrics():
    gat_csv_path = ROOT_DIR / "credit_rating_artifacts" / "gat_metrics.csv"
    if gat_csv_path.exists():
        print(f"Loading dynamic GAT metrics from {gat_csv_path}")
        gat_df = pd.read_csv(gat_csv_path)
        gat_test_row = gat_df[gat_df["Split"].str.contains("Test_Class0Calibrated", na=False)]
        if gat_test_row.empty:
            gat_test_row = gat_df[gat_df["Split"].str.contains("Test", na=False)]
        if not gat_test_row.empty:
            row = gat_test_row.iloc[0]
            return {
                "Accuracy": float(row["Accuracy"]),
                "Precision": float(row["Precision_Weighted"]),
                "Recall": float(row["Recall_Weighted"]),
                "F1-Score": float(row["Weighted_F1"]),
                "AUC": float(row["AUC"])
            }
    print("Warning: GAT metrics CSV not found. Using fallback.")
    return {
        "Accuracy": 0.909460,
        "Precision": 0.910239,
        "Recall": 0.909460,
        "F1-Score": 0.906394,
        "AUC": 0.950667
    }

def load_dmf_metrics():
    dmf_csv_path = ROOT_DIR / "credit_rating_artifacts" / "dmf_gat_lstm" / "dmf_dcs_metrics.csv"
    if dmf_csv_path.exists():
        print(f"Loading dynamic DMF metrics from {dmf_csv_path}")
        dmf_df = pd.read_csv(dmf_csv_path)
        dmf_test_row = dmf_df[dmf_df["Model"].str.contains("DMF/DCS proposed", na=False)]
        if dmf_test_row.empty:
            dmf_test_row = dmf_df[dmf_df["Model"].str.contains("constant", na=False)]
        if not dmf_test_row.empty:
            row = dmf_test_row.iloc[0]
            return {
                "Accuracy": float(row["Accuracy"]),
                "Precision": float(row["Precision_Weighted"]),
                "Recall": float(row["Recall_Weighted"]),
                "F1-Score": float(row["Weighted_F1"]),
                "AUC": float(row["AUC"])
            }
    print("Warning: DMF metrics CSV not found. Using fallback.")
    return {
        "Accuracy": 0.919327,
        "Precision": 0.919466,
        "Recall": 0.919327,
        "F1-Score": 0.915169,
        "AUC": 0.954297
    }

def update_excel():
    print(f"Reading excel sheet from {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")
    
    # Trim and clean columns
    df.columns = [str(c).strip() for c in df.columns]
    
    # Check for GAT (KB13) and DMF (KB14) and add them if not present
    if not (df["Kịch bản"] == "KB13").any():
        print("Adding GAT (KB13) row to DataFrame")
        new_row = pd.DataFrame([{
            "Kịch bản": "KB13",
            "Tên kịch bản": "GAT",
            "Phương pháp": "",
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-Score": 0.0,
            "AUC": 0.0
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        
    if not (df["Kịch bản"] == "KB14").any():
        print("Adding DMF (KB14) row to DataFrame")
        new_row = pd.DataFrame([{
            "Kịch bản": "KB14",
            "Tên kịch bản": "DMF",
            "Phương pháp": "Dynamic Model Fusion",
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-Score": 0.0,
            "AUC": 0.0
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        
    gat_metrics = load_gat_metrics()
    dmf_metrics = load_dmf_metrics()
    
    # Iterate through rows and update based on "Kịch bản" code
    for idx, row in df.iterrows():
        kb_code = str(row["Kịch bản"]).strip()
        if kb_code in updated_metrics:
            metrics = updated_metrics[kb_code]
            for metric_col, val in metrics.items():
                df.at[idx, metric_col] = val
        elif kb_code == "KB13":
            print(f"Updating KB13 (GAT) with metrics: {gat_metrics}")
            for metric_col, val in gat_metrics.items():
                df.at[idx, metric_col] = val
        elif kb_code == "KB14":
            print(f"Updating KB14 (DMF) with metrics: {dmf_metrics}")
            for metric_col, val in dmf_metrics.items():
                df.at[idx, metric_col] = val
                
    # Save the file back to excel
    df.to_excel(EXCEL_PATH, sheet_name="Sheet1", index=False)
    print("Excel updated successfully!")

def run_plotting_script():
    print(f"Running plotting script: {PLOTTING_SCRIPT}")
    result = subprocess.run([sys.executable, str(PLOTTING_SCRIPT)], capture_output=True, text=True)
    if result.returncode == 0:
        print("Plots generated successfully!")
        print(result.stdout)
    else:
        print("Failed to generate plots.")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

if __name__ == "__main__":
    update_excel()
    run_plotting_script()
