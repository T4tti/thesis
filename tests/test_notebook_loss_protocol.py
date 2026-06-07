from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = [
    "lstm-baseline.ipynb",
    "tcn-baseline.ipynb",
    "patchtst-baseline.ipynb",
    "Sparse-Graph-baseline.ipynb",
    "Transformer-LSTM.ipynb",
    "lightgbm-baseline.ipynb",
    "xgboost-baseline.ipynb",
    "kb7-fi-ttx.ipynb",
    "kb8-fi-pll.ipynb",
    "kb9-fi-ttlpxl.ipynb",
    "kb10-fr-ttx.ipynb",
    "kb11-fr-pll.ipynb",
    "kb12-fr-ttlpxl.ipynb",
    "dynamic-model-fusion.ipynb",
]
NEURAL_NOTEBOOKS = NOTEBOOKS[:5]
TREE_NOTEBOOKS = NOTEBOOKS[5:7]
ENSEMBLE_NOTEBOOKS = NOTEBOOKS[7:13]


def notebook_source(name: str) -> str:
    notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_json_ast_and_shared_contract(name: str) -> None:
    notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))

    source = notebook_source(name)
    assert "Two-tier loss source=" in source
    assert 'BENCHMARK_PROTOCOL = "benchmark_ce"' in source
    assert 'ORDINAL_PROTOCOL = "ordinal_ce_emd"' in source
    assert 'TARGET_ORDERED_LABELS = ["Distressed", "HY", "IG"]' in source
    assert "seed=42" in source or "SEED = 42" in source

    forbidden = (
        "train_augmented_timegan",
        "train_augmented_ctgan",
        "corn_logits_to_proba",
        "coral-pytorch",
        "RampedCEFocalOrdinalLoss",
        "RampedFocalOrdinalLoss",
        "FocalOrdinalLoss",
        "WeightedRandomSampler",
    )
    for term in forbidden:
        assert term not in source


@pytest.mark.parametrize("name", NEURAL_NOTEBOOKS)
def test_neural_notebook_tracks_two_tier_losses(name: str) -> None:
    source = notebook_source(name)
    assert "build_loss(" in source
    assert "train_NLL" in source
    assert "val_NLL" in source
    assert "train_Objective" in source
    assert "val_Objective" in source
    assert "LOSS_PROTOCOL == BENCHMARK_PROTOCOL" in source


@pytest.mark.parametrize("name", TREE_NOTEBOOKS)
def test_tree_notebook_selects_iteration_by_protocol(name: str) -> None:
    source = notebook_source(name)
    assert 'n_estimators=100' in source
    assert 'selection_column = "val_NLL" if LOSS_PROTOCOL == BENCHMARK_PROTOCOL else "val_Objective"' in source
    assert "best_iteration" in source


@pytest.mark.parametrize("name", ENSEMBLE_NOTEBOOKS)
def test_ensemble_uses_calibrated_inputs_and_two_tier_report(name: str) -> None:
    source = notebook_source(name)
    assert "cal_vp_path" in source
    assert "cal_tp_path" in source
    assert "numpy_objective(" in source
    assert "cross_fit_temperature_scaling(" in source
    assert "calibrated_validation_probabilities" in source
    assert "calibrated_test_probabilities" in source


def test_fusion_prefers_calibrated_probabilities_and_nll_reliability() -> None:
    source = notebook_source("dynamic-model-fusion.ipynb")
    assert "calibrated_cols" in source
    assert "reliability_weights_from_nll(" in source
    assert "DCS_MODEL_WEIGHT_MODE = 'validation_nll'" in source


@pytest.mark.parametrize(
    ("name", "val_frame", "test_frame"),
    [
        ("lstm-baseline.ipynb", "lstm_val_predictions", "lstm_test_predictions"),
        ("Sparse-Graph-baseline.ipynb", "gat_val_predictions", "gat_test_predictions"),
        ("Transformer-LSTM.ipynb", "tlstm_val_predictions", "tlstm_test_predictions"),
    ],
)
def test_fusion_csv_contract_keeps_raw_and_calibrated_probabilities(
    name: str,
    val_frame: str,
    test_frame: str,
) -> None:
    source = notebook_source(name)
    assert "prob_" in source
    assert 'f"cal_prob_{_class_id}"' in source
    assert f"{val_frame}.to_csv" in source
    assert f"{test_frame}.to_csv" in source
    for column in ("row_id", "true_label", "pred_label", "confidence"):
        assert column in source
