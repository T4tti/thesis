import ast
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
SHARED_ROW_ID = "000518"
SHARED_TICKER = "KOS"
SHARED_DATE = "2016-02-09"

CANONICAL_ROW_NOTEBOOKS = [
    "GraphSAGE-baseline.ipynb",
    "lightgbm-baseline.ipynb",
    "lstm-baseline.ipynb",
    "patchtst-baseline.ipynb",
    "tcn-baseline.ipynb",
    "xgboost-baseline.ipynb",
    "kb7-fi-ttx.ipynb",
    "kb8-fi-pll.ipynb",
    "kb9-fi-ttlpxl.ipynb",
    "kb10-fr-ttx.ipynb",
    "kb11-fr-pll.ipynb",
    "kb12-fr-ttlpxl.ipynb",
]

INDEX_518_NOTEBOOKS = [
    "dynamic-model-fusion.ipynb",
    "transformer-lstm.ipynb",
]


def _xai_cell(notebook_name: str) -> tuple[dict, str]:
    notebook = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and (
            "XAI_SHARED_ROW_ID" in source or "XAI_TEST_INDEX" in source
        ):
            return cell, source
    raise AssertionError(f"{notebook_name} has no shared-row xAI cell")


def test_raw_test_row_000518_is_the_kos_case_study() -> None:
    with (ROOT / "data" / "processed" / "test.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    row = rows[int(SHARED_ROW_ID)]
    assert row["ticker"] == SHARED_TICKER
    assert row["company_name"] == "Kosmos Energy, Ltd."
    assert row["rating_date"] == SHARED_DATE


def test_all_lime_notebooks_enforce_canonical_row_000518() -> None:
    for notebook_name in CANONICAL_ROW_NOTEBOOKS:
        cell, source = _xai_cell(notebook_name)
        ast.parse(source)

        assert f'XAI_SHARED_ROW_ID = "{SHARED_ROW_ID}"' in source
        assert f'"ticker": "{SHARED_TICKER}"' in source
        assert f'"rating_date": "{SHARED_DATE}"' in source
        assert '"model_row_id"' in source
        assert '"row_id": shared_row_id' in source
        assert "financial_lime_row_{shared_row_id}.html" in source
        assert "LIME row_id={shared_row_id} | model_row_id=" in source

        # Old executed output could display a different row even when source is correct.
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None


def test_transformer_lstm_and_dmf_use_test_index_518() -> None:
    for notebook_name in INDEX_518_NOTEBOOKS:
        cell, source = _xai_cell(notebook_name)
        ast.parse(source)

        assert "XAI_TEST_INDEX = 518" in source
        assert '"ticker": "EMN"' in source
        assert '"company_name": "EASTMAN CHEMICAL COMPANY"' in source
        assert '"rating_date": "2013-09-05"' in source
        assert "return [idx]" in source
        assert "LIME Test index={test_index}" in source
        assert "Direct LIME using notebook probability outputs" in source
        assert "financial_lime_test_idx_{idx}.html" in source
        assert '"test_index": idx' in source

        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None


def test_dmf_probability_head_helpers_are_defined_before_use() -> None:
    _, source = _xai_cell("dynamic-model-fusion.ipynb")

    required_helpers = [
        "class _XAIProbabilityHead",
        "def _xai_resolve_torch_device",
        "def _xai_train_probability_head",
        "def _xai_predict_from_head",
    ]
    for helper in required_helpers:
        assert helper in source

    definition = source.index("def _xai_train_probability_head")
    call = source.index(
        "xai_head, X_std, xai_center, xai_scale, xai_device, proxy_fidelity = "
        "_xai_train_probability_head(X_view, proba)"
    )
    assert definition < call
