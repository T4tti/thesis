import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SCRATCH_DIR = ROOT_DIR / "scratch"
ARTIFACTS_DIR = ROOT_DIR / "credit_rating_artifacts"
DMF_DIR = ARTIFACTS_DIR / "dmf_gat_lstm"

def prepare_directories():
    print("Preparing directories and copying necessary predictions...")
    DMF_DIR.mkdir(parents=True, exist_ok=True)
    
    # GAT/dmf_gat_lstm has pre-computed predictions for LSTM (needed for DMF)
    src_dir = ROOT_DIR / "artifacts" / "GAT" / "dmf_gat_lstm"
    if src_dir.exists():
        for filename in ["lstm_val_predictions.csv", "lstm_test_predictions.csv", "label_mapping.csv"]:
            src_file = src_dir / filename
            dst_file = DMF_DIR / filename
            if src_file.exists() and not dst_file.exists():
                print(f"Copying {filename} to {dst_file}")
                shutil.copy(src_file, dst_file)
    else:
        print("Warning: Source predictions directory artifacts/GAT/dmf_gat_lstm not found!")

def ipynb_to_py(nb_path: Path, py_path: Path):
    print(f"Converting {nb_path.name} to {py_path.name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_lines = [
        "import matplotlib",
        "matplotlib.use('Agg')  # Headless plotting",
        "import sys",
        "sys.stdout.reconfigure(encoding='utf-8')  # Avoid character encoding issues",
        "def display(*args, **kwargs):",
        "    for arg in args:",
        "        print(arg)",
        ""
    ]
    
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        
        # Clean magic commands and shell execution
        cleaned_source = []
        for line in cell.get("source", []):
            if line.strip().startswith("%") or line.strip().startswith("!"):
                cleaned_source.append("# " + line)
            else:
                cleaned_source.append(line)
        
        code_lines.extend(cleaned_source)
        code_lines.append("\n# " + "-"*40 + "\n")
        
    with open(py_path, "w", encoding="utf-8") as f:
        f.write("\n".join(code_lines))

def execute_script(py_path: Path):
    print(f"Executing {py_path.name}...")
    result = subprocess.run([sys.executable, str(py_path)], capture_output=True, text=True, encoding="utf-8")
    if result.returncode == 0:
        print(f"Execution of {py_path.name} completed successfully.")
        print(result.stdout[-1000:])  # Print last 1000 characters of stdout
    else:
        print(f"Failed executing {py_path.name}:")
        print("STDOUT:")
        print(result.stdout[-2000:])
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError(f"Execution failed for {py_path.name}")

if __name__ == "__main__":
    prepare_directories()
    
    gat_py = SCRATCH_DIR / "temp_gat.py"
    dmf_py = SCRATCH_DIR / "temp_dmf.py"
    
    # 1. Convert and run GAT baseline
    ipynb_to_py(NOTEBOOKS_DIR / "gat-baseline.ipynb", gat_py)
    execute_script(gat_py)
    
    # 2. Copy the newly generated GAT predictions to DMF directory (just in case they were generated in a different artifact path)
    # The GAT notebook writes to credit_rating_artifacts/dmf_gat_lstm/
    
    # 3. Convert and run DMF ensemble
    ipynb_to_py(NOTEBOOKS_DIR / "dynamic-model-fusion.ipynb", dmf_py)
    execute_script(dmf_py)
    
    print("\nNotebooks executed successfully!")
