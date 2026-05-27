# ============================================================
# xAI SHAP Interpretation - Financial Feature Surrogate
# ============================================================
SHAP_FINANCIAL_ENABLED = True
XAI_MODEL_KEY = 'patchtst'
XAI_MODEL_LABEL = 'PatchTST'
XAI_SURROGATE_VIEW = 'sequence_window'
SHAP_RANDOM_STATE = SEED if 'SEED' in globals() else 42
SHAP_BACKGROUND_SIZE = 80
SHAP_MAX_SAMPLES = 80
SHAP_NSAMPLES = 200


def _xai_class_names():
    id_to_raw_local = {v: k for k, v in raw_to_id.items()} if 'raw_to_id' in globals() else {i: str(i) for i in range(n_classes)}
    return [str(id_to_raw_local.get(i, i)) for i in range(n_classes)]


def _xai_normalize_proba(pred):
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    pred = np.clip(pred, 1e-9, 1.0)
    return pred / pred.sum(axis=1, keepdims=True)


def _xai_logits_to_proba(logits):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.shape[1] == n_classes - 1 and 'corn_logits_to_proba' in globals():
        return corn_logits_to_proba(logits, n_classes=n_classes).detach().cpu().numpy()
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def _xai_sequence_financial_matrix(samples, reducer='mean'):
    if len(samples) == 0:
        raise ValueError('No sequence samples available for xAI.')
    seq = np.stack([s[0] for s in samples], axis=0).astype(np.float32)
    if reducer == 'last':
        return seq[:, -1, :]
    return seq.mean(axis=1)


def _xai_tabular_financial_matrix(X_arr, reducer='mean'):
    if 'MODEL_FEATURES' not in globals() or 'INPUT_SIZE' not in globals():
        raise RuntimeError('Missing MODEL_FEATURES/INPUT_SIZE for tabular financial xAI view.')
    X_arr = np.asarray(X_arr, dtype=np.float32)
    n_base = len(MODEL_FEATURES)
    window_width = int(INPUT_SIZE) * n_base
    if X_arr.shape[1] < window_width:
        raise ValueError(f'X has {X_arr.shape[1]} columns, expected at least {window_width} window columns.')
    windows = X_arr[:, :window_width].reshape(X_arr.shape[0], int(INPUT_SIZE), n_base)
    if reducer == 'last':
        return windows[:, -1, :]
    return windows.mean(axis=1)


def _xai_predict_sequence_proba(samples, batch_size=256):
    model.eval()
    probas = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        xb = torch.tensor(np.stack([s[0] for s in batch], axis=0), dtype=torch.float32, device=device)
        lyb = torch.tensor([s[1] for s in batch], dtype=torch.long, device=device)
        sb = torch.tensor([s[2] for s in batch], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(xb, lyb, sb)
            probas.append(_xai_logits_to_proba(logits))
    return np.vstack(probas)


def _xai_predict_transformer_proba(samples, batch_size=256):
    model.eval()
    probas = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        xb = torch.tensor(np.stack([s[0] for s in batch], axis=0), dtype=torch.float32, device=device)
        lyb = torch.tensor([s[1] for s in batch], dtype=torch.long, device=device)
        sb = torch.tensor([s[2] for s in batch], dtype=torch.long, device=device)
        tb = torch.tensor([s[3] for s in batch], dtype=torch.long, device=device)
        cb = torch.tensor([s[4] for s in batch], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(xb, lyb, sb, tb, cb, return_aux=False)
            probas.append(_xai_logits_to_proba(logits))
    return np.vstack(probas)


def _prepare_financial_surrogate():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error

    if 'MODEL_FEATURES' not in globals():
        raise RuntimeError('Missing MODEL_FEATURES. Run preprocessing/training cells first.')

    if XAI_SURROGATE_VIEW == 'tabular_window':
        if 'model' not in globals() or 'X_train' not in globals() or 'X_test' not in globals():
            raise RuntimeError('Missing model/X_train/X_test. Run training cells first.')
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
        X_train_fin = _xai_tabular_financial_matrix(X_train_arr, reducer='mean')
        X_test_fin = _xai_tabular_financial_matrix(X_test_arr, reducer='mean')
        train_model_proba = _xai_normalize_proba(model.predict_proba(X_train_arr))
        test_model_proba = _xai_normalize_proba(model.predict_proba(X_test_arr))
    elif XAI_SURROGATE_VIEW == 'sequence_window':
        if 'model' not in globals() or 'train_ds' not in globals() or 'test_ds' not in globals():
            raise RuntimeError('Missing model/train_ds/test_ds. Run training cells first.')
        train_samples = train_ds.samples
        test_samples = test_ds.samples
        X_train_fin = _xai_sequence_financial_matrix(train_samples, reducer='mean')
        X_test_fin = _xai_sequence_financial_matrix(test_samples, reducer='mean')
        train_model_proba = _xai_predict_sequence_proba(train_samples)
        test_model_proba = _xai_predict_sequence_proba(test_samples)
    elif XAI_SURROGATE_VIEW == 'transformer_window':
        if 'model' not in globals() or 'train_seqs' not in globals() or 'test_seqs' not in globals():
            raise RuntimeError('Missing model/train_seqs/test_seqs. Run training cells first.')
        X_train_fin = _xai_sequence_financial_matrix(train_seqs, reducer='mean')
        X_test_fin = _xai_sequence_financial_matrix(test_seqs, reducer='mean')
        train_model_proba = _xai_predict_transformer_proba(train_seqs)
        test_model_proba = _xai_predict_transformer_proba(test_seqs)
    else:
        raise ValueError(f'Unsupported XAI_SURROGATE_VIEW={XAI_SURROGATE_VIEW}')

    surrogate = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=5,
        random_state=SHAP_RANDOM_STATE,
        n_jobs=-1,
    )
    surrogate.fit(X_train_fin, train_model_proba)

    def predict_financial_surrogate(x_batch):
        x_batch = np.asarray(x_batch, dtype=np.float64)
        if x_batch.ndim == 1:
            x_batch = x_batch.reshape(1, -1)
        return _xai_normalize_proba(surrogate.predict(x_batch))

    train_surrogate_proba = predict_financial_surrogate(X_train_fin)
    test_surrogate_proba = predict_financial_surrogate(X_test_fin)
    train_fidelity = accuracy_score(np.argmax(train_model_proba, axis=1), np.argmax(train_surrogate_proba, axis=1))
    test_fidelity = accuracy_score(np.argmax(test_model_proba, axis=1), np.argmax(test_surrogate_proba, axis=1))
    train_mse = mean_squared_error(train_model_proba, train_surrogate_proba)
    test_mse = mean_squared_error(test_model_proba, test_surrogate_proba)

    print(f'[INFO] Financial-feature surrogate fitted for {XAI_MODEL_LABEL} using {XAI_SURROGATE_VIEW}.')
    print(f'[INFO] Features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}')
    print(f'[INFO] Surrogate fidelity vs model | Train acc={train_fidelity:.4f}, mse={train_mse:.6f} | Test acc={test_fidelity:.4f}, mse={test_mse:.6f}')
    print('[WARN] LIME/SHAP explain this financial-feature surrogate, not the full internal model directly.')

    return {
        'feature_cols': list(MODEL_FEATURES),
        'X_train_fin': X_train_fin,
        'X_test_fin': X_test_fin,
        'train_model_proba': train_model_proba,
        'test_model_proba': test_model_proba,
        'surrogate': surrogate,
        'predict_fn': predict_financial_surrogate,
        'train_fidelity': train_fidelity,
        'test_fidelity': test_fidelity,
        'train_mse': train_mse,
        'test_mse': test_mse,
    }


if SHAP_FINANCIAL_ENABLED:
    import subprocess
    try:
        import shap
    except ImportError:
        print('Installing shap package...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap', '-q'])
        import shap

    financial_surrogate_ctx = _prepare_financial_surrogate()
    rng = np.random.default_rng(SHAP_RANDOM_STATE)
    X_train_fin = financial_surrogate_ctx['X_train_fin']
    X_test_fin = financial_surrogate_ctx['X_test_fin']
    feature_names = financial_surrogate_ctx['feature_cols']

    background_size = min(SHAP_BACKGROUND_SIZE, len(X_train_fin))
    explain_size = min(SHAP_MAX_SAMPLES, len(X_test_fin))
    background_idx = rng.choice(len(X_train_fin), size=background_size, replace=False)
    shap_indices = rng.choice(len(X_test_fin), size=explain_size, replace=False)
    background_data = X_train_fin[background_idx]
    explain_data = X_test_fin[shap_indices]

    print('Initializing SHAP KernelExplainer on financial-feature surrogate...')
    print(f'Background rows={len(background_data)}, explained test rows={len(explain_data)}')
    explainer = shap.KernelExplainer(financial_surrogate_ctx['predict_fn'], background_data)
    shap_values_raw = explainer.shap_values(explain_data, nsamples=SHAP_NSAMPLES)

    def _as_multiclass_shap_list(values, n_outputs):
        if isinstance(values, list):
            return values
        arr = np.asarray(values)
        if arr.ndim == 3:
            if arr.shape[0] == n_outputs:
                return [arr[i] for i in range(n_outputs)]
            if arr.shape[-1] == n_outputs:
                return [arr[:, :, i] for i in range(n_outputs)]
        return values

    shap_values_plot = _as_multiclass_shap_list(shap_values_raw, n_classes)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_plot,
        explain_data,
        feature_names=feature_names,
        class_names=_xai_class_names(),
        show=False,
    )
    plt.title(f'{XAI_MODEL_LABEL} Financial Feature SHAP Summary (Surrogate)', fontweight='bold')
    plt.tight_layout()
    shap_plot_path = ARTIFACT_DIR / f'{XAI_MODEL_KEY}_financial_shap_summary.png'
    plt.savefig(shap_plot_path, dpi=180, bbox_inches='tight')
    plt.show()

    if isinstance(shap_values_plot, list):
        mean_abs = np.mean([np.mean(np.abs(v), axis=0) for v in shap_values_plot], axis=0)
    else:
        arr = np.asarray(shap_values_plot)
        if arr.ndim == 3:
            mean_abs = np.mean(np.abs(arr), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(arr), axis=0)

    shap_global = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': np.ravel(mean_abs)[:len(feature_names)]})
    shap_global = shap_global.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_feature_path = ARTIFACT_DIR / f'{XAI_MODEL_KEY}_financial_shap_importance.csv'
    shap_global.to_csv(shap_feature_path, index=False)


    def _xai_pred_class_shap_vector(values, class_idx, sample_pos):
        if isinstance(values, list):
            return np.asarray(values[int(class_idx)])[sample_pos]
        arr = np.asarray(values)
        if arr.ndim == 2:
            return arr[sample_pos]
        if arr.ndim == 3:
            if arr.shape[0] == n_classes:
                return arr[int(class_idx), sample_pos, :]
            if arr.shape[-1] == n_classes:
                return arr[sample_pos, :, int(class_idx)]
        raise ValueError(f'Unsupported local SHAP output shape: {arr.shape}')

    local_rows = []
    class_names_local = _xai_class_names()
    for local_pos, test_idx in enumerate(shap_indices):
        test_idx = int(test_idx)
        pred_class = int(np.argmax(financial_surrogate_ctx['test_model_proba'][test_idx]))
        pred_prob = float(financial_surrogate_ctx['test_model_proba'][test_idx, pred_class])
        shap_vec = np.asarray(_xai_pred_class_shap_vector(shap_values_plot, pred_class, local_pos), dtype=float)
        feature_values = np.asarray(explain_data[local_pos], dtype=float)
        order = np.argsort(-np.abs(shap_vec))[:min(10, len(feature_names))]

        print(f"\n[SHAP Local Decision] test_index={test_idx} | predicted_class={class_names_local[pred_class]} | proba={pred_prob:.4f}")
        print('  Supports predicted class:')
        support_order = [j for j in order if shap_vec[j] > 0][:5]
        for j in support_order:
            print(f'    + {feature_names[j]}={feature_values[j]:.4f} | shap={shap_vec[j]:+.5f}')
        print('  Opposes predicted class:')
        oppose_order = [j for j in order if shap_vec[j] < 0][:5]
        for j in oppose_order:
            print(f'    - {feature_names[j]}={feature_values[j]:.4f} | shap={shap_vec[j]:+.5f}')

        for j in order:
            local_rows.append({
                'test_index': test_idx,
                'predicted_class': class_names_local[pred_class],
                'predicted_class_id': pred_class,
                'predicted_probability': pred_prob,
                'feature': feature_names[j],
                'feature_value': float(feature_values[j]),
                'shap_value_for_predicted_class': float(shap_vec[j]),
                'direction': 'supports_predicted_class' if shap_vec[j] > 0 else 'opposes_predicted_class',
                'abs_shap': float(abs(shap_vec[j])),
            })

    shap_local_path = ARTIFACT_DIR / f'{XAI_MODEL_KEY}_financial_shap_local_decisions.csv'
    shap_local_df = pd.DataFrame(local_rows)
    shap_local_df.to_csv(shap_local_path, index=False)
    print(f'Saved local SHAP decision explanations to: {shap_local_path}')


    print(f'SHAP financial surrogate explained samples: {explain_size}')
    print('Top SHAP financial features:')
    print(shap_global.head(12).to_string(index=False))
    print(f'Saved: {shap_feature_path}')
    print(f'Saved: {shap_plot_path}')
else:
    print('SHAP financial xAI is disabled. Set SHAP_FINANCIAL_ENABLED=True to run explanations.')
