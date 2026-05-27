# ============================================================
# xAI LIME Interpretation - Financial Feature Surrogate
# ============================================================
LIME_FINANCIAL_ENABLED = True
XAI_MODEL_KEY = 'patchtst'
XAI_MODEL_LABEL = 'PatchTST'
XAI_SURROGATE_VIEW = 'sequence_window'
LIME_RANDOM_STATE = SEED if 'SEED' in globals() else 42

# This cell uses the same financial surrogate contract as the SHAP cell.
# If the SHAP cell was not run, it prepares the surrogate here.
if LIME_FINANCIAL_ENABLED:
    import subprocess
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        print('Installing lime package...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lime', '-q'])
        import lime
        import lime.lime_tabular

    if '_prepare_financial_surrogate' not in globals():
        raise RuntimeError('Run the financial SHAP/setup cell before running LIME.')
    if 'financial_surrogate_ctx' not in globals():
        financial_surrogate_ctx = _prepare_financial_surrogate()

    lime_class_names = _xai_class_names()
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=financial_surrogate_ctx['X_train_fin'],
        feature_names=financial_surrogate_ctx['feature_cols'],
        class_names=lime_class_names,
        mode='classification',
        discretize_continuous=True,
        random_state=LIME_RANDOM_STATE,
    )

    rng = np.random.default_rng(LIME_RANDOM_STATE)
    X_test_fin = financial_surrogate_ctx['X_test_fin']
    selected_indices = rng.choice(len(X_test_fin), size=min(3, len(X_test_fin)), replace=False)

    print(f'LIME financial-feature surrogate explanations for {len(selected_indices)} test samples:')
    lime_local_rows = []
    for idx in selected_indices:
        model_pred = int(np.argmax(financial_surrogate_ctx['test_model_proba'][idx]))
        print(f"\n--- Test index {idx} | {XAI_MODEL_LABEL} predicted class={lime_class_names[model_pred]} ---")
        exp = explainer.explain_instance(
            data_row=X_test_fin[idx],
            predict_fn=financial_surrogate_ctx['predict_fn'],
            num_features=min(10, len(financial_surrogate_ctx['feature_cols'])),
            labels=[model_pred],
        )
        try:
            exp.show_in_notebook(show_table=True)
        except Exception:
            pass

        html_path = ARTIFACT_DIR / f'{XAI_MODEL_KEY}_financial_lime_test_idx_{idx}.html'
        exp.save_to_file(str(html_path))
        print(f'Saved financial LIME explanation HTML to: {html_path}')
        print('LIME Financial Feature Explanation Details:')

        lime_items = exp.as_list(label=model_pred)
        print('Why this class?')
        for feature, weight in lime_items:
            direction = 'supports predicted class' if weight > 0 else 'opposes predicted class'
            print(f'  {direction:<25} | {feature:<45} | weight={weight:+.4f}')
            lime_local_rows.append({
                'test_index': int(idx),
                'predicted_class': lime_class_names[model_pred],
                'predicted_class_id': int(model_pred),
                'predicted_probability': float(financial_surrogate_ctx['test_model_proba'][idx, model_pred]),
                'lime_rule': feature,
                'lime_weight_for_predicted_class': float(weight),
                'direction': 'supports_predicted_class' if weight > 0 else 'opposes_predicted_class',
            })

    lime_local_path = ARTIFACT_DIR / f'{XAI_MODEL_KEY}_financial_lime_local_decisions.csv'
    lime_local_df = pd.DataFrame(lime_local_rows)
    lime_local_df.to_csv(lime_local_path, index=False)
    print(f'Saved local LIME decision explanations to: {lime_local_path}')
else:
    print('LIME financial xAI is disabled. Set LIME_FINANCIAL_ENABLED=True to run explanations.')
