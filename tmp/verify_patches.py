"""Verify all patches were applied to the notebook."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "transformer-lstm.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))
src = "".join([line for cell in nb["cells"] for line in cell.get("source", [])])

checks = [
    ("BATCH_SIZE = 64", "Batch size 32→64"),
    ("TRAIN_WINDOW_NOISE_STD = 0.01", "Noise 0.02→0.01"),
    ("TRAIN_FEATURE_DROPOUT = 0.03", "Feature dropout 0.05→0.03"),
    ("'dropout': 0.12", "Model dropout 0.20→0.12"),
    ("'d_model': 128", "d_model 96→128"),
    ("'n_layers': 3", "n_layers 2→3"),
    ("'hidden_size': 128", "hidden_size 96→128"),
    ("'max_lr': 5.0e-4", "max_lr 8e-4→5e-4"),
    ("MIXUP_PROB = 0.0", "Mixup disabled"),
    ("LAST_Y_CONTEXT_DROPOUT_MAX = 0.10", "Context dropout 0.20→0.10"),
    ("LAST_Y_CONTEXT_PERMUTE_MAX = 0.03", "Context permute 0.08→0.03"),
    ("TRANSITION_PENALTY_WEIGHT_MAX = 0.12", "TransPen 0.22→0.12"),
    ("TRANSITION_WARMUP_FRACTION = 0.30", "TransWarmup 0.15→0.30"),
    ("LinearLR", "Linear warmup scheduler"),
    ("CosineAnnealingLR", "Cosine annealing scheduler"),
    ("clip_grad_norm_(model.parameters(), 1.5)", "Grad clip 0.8→1.5"),
    ("logits = raw_logits", "No blend during training"),
    ("current_metric = float(current_metric_raw)", "No EMA early stopping"),
]

ok = 0
fail = 0
for pattern, desc in checks:
    found = pattern in src
    if found:
        ok += 1
        print(f"  OK  {desc}")
    else:
        fail += 1
        print(f"  FAIL {desc}: '{pattern}' not found")

print(f"\nResult: {ok}/{ok+fail} patches verified")
if fail > 0:
    print("WARNING: Some patches were not applied!")
else:
    print("All patches applied successfully!")
