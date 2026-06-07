"""Model components for credit rating prediction."""

from .losses import (  # noqa: F401
	BENCHMARK_PROTOCOL,
	DEFAULT_LABEL_ORDER,
	DEFAULT_ORDINAL_LAMBDA,
	ORDINAL_PROTOCOL,
	TwoTierClassificationLoss,
	apply_temperature,
	build_loss,
	cdf_emd2,
	cross_fit_temperature_scaling,
	fit_temperature,
	numpy_cdf_emd2,
	numpy_nll,
	numpy_objective,
	probability_report,
	reliability_weights_from_nll,
)

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
