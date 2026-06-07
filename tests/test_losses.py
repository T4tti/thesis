from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss

from src.models.losses import (
    BENCHMARK_PROTOCOL,
    ORDINAL_PROTOCOL,
    apply_temperature,
    benchmark_ce,
    cross_fit_temperature_scaling,
    numpy_cdf_emd2,
    numpy_nll,
    numpy_objective,
    probability_report,
)


def test_torch_ce_matches_sklearn_log_loss() -> None:
    logits = torch.tensor(
        [[2.0, 0.4, -1.0], [-0.5, 0.2, 1.7], [0.1, 1.3, -0.2]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 2, 1])
    probabilities = torch.softmax(logits, dim=1).numpy()

    expected = log_loss(targets.numpy(), probabilities, labels=[0, 1, 2])

    assert np.isclose(benchmark_ce(logits, targets).item(), expected, atol=1e-7)
    assert np.isclose(numpy_nll(probabilities, targets.numpy()), expected, atol=1e-7)


def test_cdf_emd_is_zero_for_exact_predictions() -> None:
    probabilities = np.eye(3, dtype=np.float64)
    targets = np.array([0, 1, 2])

    assert np.isclose(numpy_cdf_emd2(probabilities, targets), 0.0, atol=1e-10)


def test_cdf_emd_increases_with_ordinal_distance() -> None:
    target = np.array([0])
    adjacent = np.array([[0.0, 1.0, 0.0]])
    distant = np.array([[0.0, 0.0, 1.0]])

    assert numpy_cdf_emd2(distant, target) > numpy_cdf_emd2(adjacent, target)


def test_two_tier_objective_contract() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    targets = np.array([0, 2])
    nll = numpy_nll(probabilities, targets)

    assert np.isclose(
        numpy_objective(probabilities, targets, protocol=BENCHMARK_PROTOCOL),
        nll,
    )
    assert (
        numpy_objective(probabilities, targets, protocol=ORDINAL_PROTOCOL)
        >= nll
    )


def test_temperature_scaling_preserves_probability_contract() -> None:
    probabilities = np.array(
        [[0.90, 0.08, 0.02], [0.70, 0.20, 0.10], [0.05, 0.15, 0.80]]
    )
    scaled = apply_temperature(probabilities, 1.7)

    assert scaled.shape == probabilities.shape
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert np.all(scaled > 0.0)


def test_cross_fitted_temperature_and_report() -> None:
    validation_targets = np.tile(np.arange(3), 4)
    validation_probabilities = np.full((12, 3), 0.10)
    validation_probabilities[np.arange(12), validation_targets] = 0.80
    validation_probabilities[
        np.arange(12),
        (validation_targets + 1) % 3,
    ] = 0.10
    test_probabilities = validation_probabilities[:5]

    calibrated = cross_fit_temperature_scaling(
        validation_probabilities,
        validation_targets,
        test_probabilities,
        max_splits=5,
        seed=42,
    )
    report = probability_report(
        validation_targets,
        calibrated.validation_probabilities,
        protocol=ORDINAL_PROTOCOL,
    )

    assert calibrated.n_splits == 4
    assert calibrated.temperature > 0.0
    assert np.allclose(calibrated.validation_probabilities.sum(axis=1), 1.0)
    assert np.allclose(calibrated.test_probabilities.sum(axis=1), 1.0)
    assert report["Protocol"] == ORDINAL_PROTOCOL
    assert report["Objective"] >= report["NLL"]


def test_notebook_fallback_matches_shared_module() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "lstm-baseline.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    bootstrap = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "two-tier-loss-contract" in cell.get("metadata", {}).get("tags", [])
    )
    tree = ast.parse(bootstrap)
    fallback_try = next(node for node in tree.body if isinstance(node, ast.Try))
    setup_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.lineno < fallback_try.lineno
    ]
    fallback_module = ast.fix_missing_locations(
        ast.Module(
            body=[*setup_nodes, *fallback_try.handlers[0].body],
            type_ignores=[],
        )
    )
    namespace: dict[str, object] = {}
    exec(compile(fallback_module, str(notebook_path), "exec"), namespace)

    probabilities = np.array(
        [[0.72, 0.20, 0.08], [0.10, 0.30, 0.60], [0.15, 0.70, 0.15]]
    )
    targets = np.array([0, 2, 1])

    assert np.isclose(namespace["numpy_nll"](probabilities, targets), numpy_nll(probabilities, targets))
    assert np.isclose(
        namespace["numpy_cdf_emd2"](probabilities, targets),
        numpy_cdf_emd2(probabilities, targets),
    )
    assert np.isclose(
        namespace["numpy_objective"](
            probabilities,
            targets,
            protocol=ORDINAL_PROTOCOL,
            ordinal_lambda=0.10,
        ),
        numpy_objective(
            probabilities,
            targets,
            protocol=ORDINAL_PROTOCOL,
            ordinal_lambda=0.10,
        ),
    )
    assert np.allclose(
        namespace["apply_temperature"](probabilities, 1.4),
        apply_temperature(probabilities, 1.4),
    )
