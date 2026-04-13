"""Model components for credit rating prediction."""

try:  # Optional dependency: torch_geometric is required for HHGNN modules.
	from .hhgnn_fuzzy import (  # noqa: F401
		DEFAULT_FINANCIAL_FEATURES,
		FuzzyFocalLoss,
		HHGNNFuzzyClassifier,
		StaticHHGNNDataset,
		build_feature_graph_dataset,
		build_fuzzy_feature_graph,
		classification_metrics,
		compute_effective_class_weights,
		compute_fuzzy_sample_weights,
		prepare_static_company_dataset,
		set_seed,
	)
except Exception:
	# Keep package importable for non-graph pipelines.
	pass
