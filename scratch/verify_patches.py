"""Verify key patches in the notebook."""
import json

nb_path = 'notebooks/Transformer-LSTM.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")

# Verify key values
checks = [
    (16, "INPUT_SIZE_DEFAULT = 8", "INPUT_SIZE"),
    (23, "MODEL_D_MODEL = 128", "d_model"),
    (23, "TRANSFORMER_HEADS = 8", "heads"),
    (23, "TRANSFORMER_LAYERS = 2", "layers"),
    (23, "LSTM_HIDDEN = 128", "hidden"),
    (23, "TICKER_EMB_DIM = 16", "ticker_emb"),
    (23, "COMPANY_EMB_DIM = 16", "company_emb"),
    (23, "TLSTM_DROPOUT = 0.35", "dropout"),
    (23, "use_transition_head=False", "transition_head conditional"),
    (23, "if self.use_transition_head and return_aux:", "transition_head forward guard"),
    (24, "'val_accuracy': 0.65", "PriMO accuracy weight"),
    (24, "PRIMO_TRIALS = 24", "PriMO trials"),
    (24, "PRIMO_FIDELITY_EPOCHS = 15", "PriMO fidelity"),
    (24, "PRIMO_TRAIN_FRACTION = 0.70", "PriMO train fraction"),
    (24, "'hidden_size': 128", "PriMO default hidden"),
    (24, "'prior_name': 'high_capacity_bilstm'", "BiLSTM prior"),
    (27, "EARLY_STOP_METRIC = 'val_accuracy'", "early-stop metric"),
    (27, "current_metric = float(vl_acc)", "checkpoint selection"),
    (27, "MIN_LABEL_SMOOTHING_FOR_GENERALIZATION = 0.0", "label_smoothing floor"),
    (27, "SWA_ENABLED = True", "SWA enabled"),
    (27, "swa_model = AveragedModel(swa_base_model)", "SWA model creation"),
    (30, "SEED_ENSEMBLE_TOTAL_RUNS = 5", "ensemble runs"),
    (30, "SEED_ENSEMBLE_TOP_K = 3", "ensemble top_k"),
    (30, "0.70 * metrics['accuracy']", "ensemble acc weight"),
]

passed = 0
failed = 0
for cell_idx, pattern, name in checks:
    src = ''.join(cells[cell_idx]['source']) if isinstance(cells[cell_idx]['source'], list) else cells[cell_idx]['source']
    if pattern in src:
        print(f"  PASS: [{cell_idx:2d}] {name}")
        passed += 1
    else:
        print(f"  FAIL: [{cell_idx:2d}] {name} -- pattern not found: {pattern[:60]}")
        failed += 1

print(f"\nResults: {passed} passed, {failed} failed out of {len(checks)}")
