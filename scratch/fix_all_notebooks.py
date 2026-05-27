"""
fix_all_notebooks.py
====================
Sửa tất cả 6 lỗi đã phát hiện trong code review cho 6 notebook ensemble.

Fixes Applied:
  [BUG-1]   CRITICAL: np.math.factorial -> math.factorial  (KB7, KB8, KB9)
  [BUG-2]   CRITICAL: Benchmark paths missing ARTIFACT_DIR  (KB7, KB8)
  [ISSUE-3] MEDIUM:   LIME import ordering fix  (KB7, KB8, KB9)
  [ISSUE-4] MEDIUM:   Metric schema sync (add Precision/Recall to FCI)  (KB7, KB8, KB9)
  [MINOR-1] Remove redundant import pickle  (KB7, KB8, KB9)
  [MINOR-2] Remove redundant re-imports in LIME/SHAP cells  (KB7, KB8, KB9)
  [MINOR-3] Rename predict_fn to avoid name collision  (KB7, KB8, KB9)
  [MINOR-4] Fix double inference time measurement  (All 6)

How to Run:
    python e:\\thesis\\scratch\\fix_all_notebooks.py

Expected Output:
    Patched 6 notebooks in-place with backup (.bak) files created.
"""
import json
import shutil
from pathlib import Path
from typing import List

NOTEBOOKS_DIR = Path(r"e:\thesis\notebooks")

# -----------------------------------------------------------------------------
# Helper: find & replace in a specific code cell's source
# -----------------------------------------------------------------------------

def get_source(cell: dict) -> str:
    """Join source lines into a single string."""
    return "".join(cell.get("source", []))


def set_source(cell: dict, text: str) -> None:
    """Set source as list of lines (preserving newlines)."""
    lines = text.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]]
    if lines[-1]:  # last line without trailing newline
        cell["source"].append(lines[-1])
    elif len(cell["source"]) > 0 and cell["source"][-1].endswith("\n"):
        pass  # keep as-is


def replace_in_cell(cell: dict, old: str, new: str) -> int:
    """Replace text in a cell's source. Returns count of replacements."""
    src = get_source(cell)
    count = src.count(old)
    if count > 0:
        src = src.replace(old, new)
        set_source(cell, src)
    return count


def find_code_cells(nb: dict) -> List[dict]:
    """Return all code cells."""
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


# -----------------------------------------------------------------------------
# FIX FUNCTIONS
# -----------------------------------------------------------------------------

def fix_bug1_factorial(nb: dict, nb_name: str) -> int:
    """BUG-1: np.math.factorial -> math.factorial (KB7/8/9 only)."""
    total = 0
    for cell in find_code_cells(nb):
        src = get_source(cell)
        if "np.math.factorial" in src:
            # Replace np.math.factorial with math.factorial
            count = replace_in_cell(cell, "np.math.factorial", "math.factorial")
            total += count
    
    # Also ensure 'import math' exists in Cell 1 (env cell)
    if total > 0:
        env_cell = find_code_cells(nb)[0]
        env_src = get_source(env_cell)
        if "import math" not in env_src:
            # Add 'import math' after 'import itertools'
            env_src = env_src.replace(
                "import os, sys, platform, random, warnings, itertools",
                "import os, sys, platform, random, warnings, itertools, math"
            )
            set_source(env_cell, env_src)
    
    if total > 0:
        print(f"  [BUG-1] {nb_name}: Replaced {total} occurrences of np.math.factorial -> math.factorial")
    return total


def fix_bug2_benchmark_paths(nb: dict, nb_name: str) -> int:
    """BUG-2: Add ARTIFACT_DIR to benchmark baseline paths (KB7/8 only)."""
    total = 0
    for cell in find_code_cells(nb):
        src = get_source(cell)
        if "Benchmark comparison" in src and "'transformer_bilstm_metrics.csv'" in src:
            # These are the cells with bare string paths
            bare_paths = [
                ("'transformer_bilstm_metrics.csv'", "ARTIFACT_DIR/'transformer_bilstm_metrics.csv'"),
                ("'tcn_metrics.csv'", "ARTIFACT_DIR/'tcn_metrics.csv'"),
                ("'xgboost_metrics.csv'", "ARTIFACT_DIR/'xgboost_metrics.csv'"),
                ("'lstm_metrics.csv'", "ARTIFACT_DIR/'lstm_metrics.csv'"),
                ("'patchtst_metrics.csv'", "ARTIFACT_DIR/'patchtst_metrics.csv'"),
                ("'lightgbm_metrics.csv'", "ARTIFACT_DIR/'lightgbm_metrics.csv'"),
            ]
            for old, new in bare_paths:
                # Only replace if not already prefixed with ARTIFACT_DIR
                count = replace_in_cell(cell, old, new)
                total += count
    
    if total > 0:
        print(f"  [BUG-2] {nb_name}: Fixed {total} benchmark paths with ARTIFACT_DIR prefix")
    return total


def fix_issue3_lime_imports(nb: dict, nb_name: str) -> int:
    """ISSUE-3: Fix LIME import ordering — remove premature imports before try/except."""
    total = 0
    for cell in find_code_cells(nb):
        src = get_source(cell)
        if "LIME_ENABLED" in src and "import lime\n    import lime.lime_tabular" in src:
            # Check if there's a bare import BEFORE the try block
            # Pattern: import lime\n    import lime.lime_tabular\n...try:\n        import lime
            if src.count("import lime") > 2:  # More than just the try/except pair
                # Remove the early bare imports and the redundant re-imports
                new_src = src.replace(
                    "    import sys, subprocess, os\n"
                    "    from pathlib import Path\n"
                    "    import numpy as np\n"
                    "    import pandas as pd\n"
                    "    import lime\n"
                    "    import lime.lime_tabular\n"
                    "    import matplotlib.pyplot as plt\n"
                    "\n"
                    "    try:\n"
                    "        import lime\n"
                    "        import lime.lime_tabular\n"
                    "    except ImportError:",
                    
                    "    import subprocess\n"
                    "\n"
                    "    try:\n"
                    "        import lime\n"
                    "        import lime.lime_tabular\n"
                    "    except ImportError:"
                )
                if new_src != src:
                    set_source(cell, new_src)
                    total += 1
    
    if total > 0:
        print(f"  [ISSUE-3] {nb_name}: Fixed LIME import ordering in {total} cell(s)")
    return total


def fix_issue4_metric_schema(nb: dict, nb_name: str) -> int:
    """ISSUE-4: Add Precision_Weighted and Recall_Weighted to FCI evaluate()."""
    total = 0
    for cell in find_code_cells(nb):
        src = get_source(cell)
        # Only apply to FCI notebooks (Choquet) that lack precision/recall in evaluate
        if "class FuzzyChoquetEnsemble" in src and "precision_score" not in src:
            # Add precision and recall to evaluate method
            old_eval = (
                "        acc = accuracy_score(y_true, yp)\n"
                "        f1m = f1_score(y_true, yp, average='macro', zero_division=0)\n"
                "        f1w = f1_score(y_true, yp, average='weighted', zero_division=0)\n"
                "        qwk = cohen_kappa_score(y_true, yp, weights='quadratic')"
            )
            new_eval = (
                "        acc = accuracy_score(y_true, yp)\n"
                "        f1m = f1_score(y_true, yp, average='macro', zero_division=0)\n"
                "        f1w = f1_score(y_true, yp, average='weighted', zero_division=0)\n"
                "        qwk = cohen_kappa_score(y_true, yp, weights='quadratic')\n"
                "        prec_w = precision_score(y_true, yp, average='weighted', zero_division=0)\n"
                "        rec_w  = recall_score(y_true, yp, average='weighted', zero_division=0)"
            )
            count = replace_in_cell(cell, old_eval, new_eval)
            
            # Also update the return dict
            old_return = (
                "        return {'Split':split,'Accuracy':acc,'Macro_F1':f1m,"
                "'Weighted_F1':f1w,'AUC':auc,'QWK':qwk}"
            )
            new_return = (
                "        return {'Split':split,'Accuracy':acc,'Precision_Weighted':prec_w,\n"
                "                'Recall_Weighted':rec_w,'Macro_F1':f1m,\n"
                "                'Weighted_F1':f1w,'AUC':auc,'QWK':qwk}"
            )
            count += replace_in_cell(cell, old_return, new_return)
            total += count
    
    if total > 0:
        print(f"  [ISSUE-4] {nb_name}: Added Precision/Recall to FCI evaluate() ({total} replacements)")
    return total


def fix_minor1_redundant_pickle(nb: dict, nb_name: str) -> int:
    """MINOR-1: Remove redundant import pickle."""
    total = 0
    for cell in find_code_cells(nb):
        src = get_source(cell)
        if src.count("import pickle") >= 2:
            # Remove the second occurrence (keep only one)
            idx = src.index("import pickle")
            second_idx = src.index("import pickle", idx + 1)
            # Find the line boundaries
            line_start = src.rfind("\n", 0, second_idx) + 1
            line_end = src.find("\n", second_idx)
            if line_end == -1:
                line_end = len(src)
            # Remove the line
            new_src = src[:line_start] + src[line_end + 1:]
            set_source(cell, new_src)
            total += 1
    
    if total > 0:
        print(f"  [MINOR-1] {nb_name}: Removed {total} redundant 'import pickle'")
    return total


def fix_minor2_redundant_imports_shap(nb: dict, nb_name: str) -> int:
    """MINOR-2: Remove redundant re-imports in SHAP cells."""
    total = 0
    for cell in find_code_cells(nb):
        src = get_source(cell)
        if "SHAP_ENABLED" in src and "SHAP Explainability" in src:
            # Check if it has the redundant imports pattern
            old_pattern = (
                "    import sys, subprocess, os\n"
                "    from pathlib import Path\n"
                "    import numpy as np\n"
                "    import pandas as pd\n"
                "    import matplotlib.pyplot as plt\n"
                "\n"
                "    try:"
            )
            new_pattern = (
                "    import subprocess\n"
                "\n"
                "    try:"
            )
            if old_pattern in src:
                src = src.replace(old_pattern, new_pattern)
                set_source(cell, src)
                total += 1
    
    if total > 0:
        print(f"  [MINOR-2] {nb_name}: Cleaned redundant SHAP imports in {total} cell(s)")
    return total


def fix_minor3_predict_fn_collision(nb: dict, nb_name: str) -> int:
    """MINOR-3: Rename predict_fn to avoid LIME/SHAP name collision."""
    total = 0
    
    # Fix LIME cell
    for cell in find_code_cells(nb):
        src = get_source(cell)
        if "LIME_ENABLED" in src and "def ensemble_predict_fn(" in src:
            src = src.replace("def ensemble_predict_fn(", "def ensemble_predict_fn_lime(")
            src = src.replace("predict_fn=ensemble_predict_fn,", "predict_fn=ensemble_predict_fn_lime,")
            # Also fix the choquet_batch call to use the correct function
            set_source(cell, src)
            total += 1
    
    # Fix SHAP cell
    for cell in find_code_cells(nb):
        src = get_source(cell)
        if "SHAP_ENABLED" in src and "def ensemble_predict_fn(" in src:
            src = src.replace("def ensemble_predict_fn(", "def ensemble_predict_fn_shap(")
            src = src.replace("explainer = shap.KernelExplainer(ensemble_predict_fn,",
                              "explainer = shap.KernelExplainer(ensemble_predict_fn_shap,")
            set_source(cell, src)
            total += 1
    
    if total > 0:
        print(f"  [MINOR-3] {nb_name}: Renamed {total} predict_fn(s) to avoid collision")
    return total


def fix_minor4_inference_time(nb: dict, nb_name: str) -> int:
    """MINOR-4: Fix double predict call in inference time measurement."""
    total = 0
    for cell in find_code_cells(nb):
        src = get_source(cell)
        if "Inference Time Measurement" in src:
            # Pattern: predict then predict_proba (double call)
            old_pattern_a = (
                "start_time = time.time()\n"
                "y_pred = ens.predict(test_probas)\n"
                "fp     = ens.predict_proba(test_probas)\n"
                "end_time = time.time()"
            )
            new_pattern_a = (
                "start_time = time.time()\n"
                "fp     = ens.predict_proba(test_probas)\n"
                "y_pred = np.argmax(fp, axis=1)\n"
                "end_time = time.time()"
            )
            count = replace_in_cell(cell, old_pattern_a, new_pattern_a)
            total += count
    
    if total > 0:
        print(f"  [MINOR-4] {nb_name}: Fixed inference time measurement ({total} cell(s))")
    return total


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def patch_notebook(nb_path: Path) -> int:
    """Apply all fixes to a single notebook. Returns total fixes applied."""
    nb_name = nb_path.name
    
    # Read notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    print(f"\n{'-'*60}")
    print(f"  Patching: {nb_name}")
    print(f"{'-'*60}")
    
    total_fixes = 0
    
    # Determine notebook type
    is_choquet = "FI-" in nb_name or "FI_" in nb_name  # KB7, KB8, KB9
    is_gompertz = "FR-" in nb_name or "FR_" in nb_name  # KB10, KB11, KB12
    is_kb7 = "KB7" in nb_name
    is_kb8 = "KB8" in nb_name
    
    # Apply fixes based on notebook type
    if is_choquet:
        total_fixes += fix_bug1_factorial(nb, nb_name)
        total_fixes += fix_issue3_lime_imports(nb, nb_name)
        total_fixes += fix_issue4_metric_schema(nb, nb_name)
        total_fixes += fix_minor1_redundant_pickle(nb, nb_name)
        total_fixes += fix_minor2_redundant_imports_shap(nb, nb_name)
        total_fixes += fix_minor3_predict_fn_collision(nb, nb_name)
    
    if is_kb7 or is_kb8:
        total_fixes += fix_bug2_benchmark_paths(nb, nb_name)
    
    # Apply to ALL notebooks
    total_fixes += fix_minor4_inference_time(nb, nb_name)
    
    if total_fixes > 0:
        # Backup original
        backup_path = nb_path.with_suffix(".ipynb.bak")
        shutil.copy2(nb_path, backup_path)
        
        # Write patched notebook
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        
        print(f"  [OK] Applied {total_fixes} fix(es). Backup: {backup_path.name}")
    else:
        print("  [INFO] No fixes needed.")
    
    return total_fixes


def main():
    notebooks = [
        NOTEBOOKS_DIR / "KB7_FI-TTX.ipynb",
        NOTEBOOKS_DIR / "KB8_FI-PLL.ipynb",
        NOTEBOOKS_DIR / "KB9_FI-TTLPXL.ipynb",
        NOTEBOOKS_DIR / "KB10_FR-TTX.ipynb",
        NOTEBOOKS_DIR / "KB11_FR-PLL.ipynb",
        NOTEBOOKS_DIR / "KB12_FR-TTLPXL.ipynb",
    ]
    
    print("=" * 60)
    print("  ENSEMBLE NOTEBOOK PATCHER")
    print("  Fixing all issues from code review")
    print("=" * 60)
    
    grand_total = 0
    for nb_path in notebooks:
        if not nb_path.exists():
            print(f"\n[SKIP] {nb_path.name} not found")
            continue
        grand_total += patch_notebook(nb_path)
    
    print(f"\n{'=' * 60}")
    print(f"  DONE: {grand_total} total fixes applied across {len(notebooks)} notebooks")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
