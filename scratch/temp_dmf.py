import matplotlib
matplotlib.use('Agg')  # Headless plotting
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Avoid character encoding issues
def display(*args, **kwargs):
    for arg in args:
        print(arg)

import os

from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (

    accuracy_score, f1_score, precision_score, recall_score,

    roc_auc_score, cohen_kappa_score, confusion_matrix, classification_report,

)

from sklearn.preprocessing import label_binarize



SEED = 42

np.random.seed(SEED)





def detect_kaggle_runtime() -> bool:

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '').strip():

        return True

    return Path('/kaggle/input').exists() and Path('/kaggle/working').exists()





IN_KAGGLE = detect_kaggle_runtime()





def find_project_root(start: Path) -> Path:

    for p in [start, *start.parents]:

        if (p / 'data').exists() and (p / 'src').exists():

            return p

    return start





PROJECT_ROOT = Path('/kaggle/working') if IN_KAGGLE else find_project_root(Path.cwd().resolve())

ARTIFACT_DIR = PROJECT_ROOT / 'credit_rating_artifacts'

KAGGLE_DMF_INPUT_ARTIFACT_DIR = Path('/kaggle/input/datasets/tailength/dmf-artifacts/dmf_gat_lstm')

DMF_ARTIFACT_DIR = ARTIFACT_DIR / 'dmf_gat_lstm'

DMF_INPUT_ARTIFACT_DIR = KAGGLE_DMF_INPUT_ARTIFACT_DIR if IN_KAGGLE and KAGGLE_DMF_INPUT_ARTIFACT_DIR.exists() else DMF_ARTIFACT_DIR

DMF_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print('Project root:', PROJECT_ROOT)

print('DMF input artifact dir:', DMF_INPUT_ARTIFACT_DIR)

print('DMF output artifact dir:', DMF_ARTIFACT_DIR)


# ----------------------------------------

LSTM_VAL_PATH = DMF_INPUT_ARTIFACT_DIR / 'lstm_val_predictions.csv'

LSTM_TEST_PATH = DMF_INPUT_ARTIFACT_DIR / 'lstm_test_predictions.csv'

GAT_VAL_PATH = DMF_INPUT_ARTIFACT_DIR / 'gat_val_predictions.csv'

GAT_TEST_PATH = DMF_INPUT_ARTIFACT_DIR / 'gat_test_predictions.csv'

LABEL_MAP_PATH = DMF_INPUT_ARTIFACT_DIR / 'label_mapping.csv'



required_paths = [LSTM_VAL_PATH, LSTM_TEST_PATH, GAT_VAL_PATH, GAT_TEST_PATH, LABEL_MAP_PATH]

missing = [str(p) for p in required_paths if not p.exists()]

if missing:

    raise FileNotFoundError('Thieu DMF input artifact. Hay chay notebook LSTM va GAT truoc: ' + '; '.join(missing))



lstm_val_raw = pd.read_csv(LSTM_VAL_PATH)

lstm_test_raw = pd.read_csv(LSTM_TEST_PATH)

gat_val_raw = pd.read_csv(GAT_VAL_PATH)

gat_test_raw = pd.read_csv(GAT_TEST_PATH)

label_map = pd.read_csv(LABEL_MAP_PATH)



print('LSTM val/test:', lstm_val_raw.shape, lstm_test_raw.shape)

print('GAT  val/test:', gat_val_raw.shape, gat_test_raw.shape)

display(label_map)


# ----------------------------------------

def prob_columns(frame):

    cols = [c for c in frame.columns if c.startswith('prob_')]

    return sorted(cols, key=lambda x: int(x.split('_')[1]))





def validate_prediction_frame(frame, model_key, split_name):

    probs = prob_columns(frame)

    if not probs:

        raise ValueError(f'{model_key} {split_name} prediction file has no prob_* columns')

    required = {'row_id', 'true_label', 'pred_label', *probs}

    missing = required - set(frame.columns)

    if missing:

        raise ValueError(f'{model_key} {split_name} prediction file missing columns: {sorted(missing)}')

    duplicated = frame.loc[frame['row_id'].duplicated(), 'row_id'].head().tolist()

    if duplicated:

        raise ValueError(f'{model_key} {split_name} has duplicate row_id values: {duplicated}')

    return probs





def standardize_predictions(frame, model_key, split_name):

    frame = frame.copy()

    probs = validate_prediction_frame(frame, model_key, split_name)

    keep_meta = [c for c in ['row_id', 'split', 'ticker', 'company_name', 'rating_date', 'true_label', 'true_label_name'] if c in frame.columns]

    out = frame[keep_meta].copy()

    out[f'{model_key}_pred_label'] = frame['pred_label'].astype(int)

    out[f'{model_key}_confidence'] = frame['confidence'].astype(float) if 'confidence' in frame.columns else frame[probs].max(axis=1).astype(float)

    for c in probs:

        out[f'{model_key}_{c}'] = frame[c].astype(float)

    return out, probs





def align_pair(lstm_frame, gat_frame, split_name):

    lstm, lstm_probs = standardize_predictions(lstm_frame, 'lstm', split_name)

    gat, gat_probs = standardize_predictions(gat_frame, 'gat', split_name)

    if lstm_probs != gat_probs:

        raise ValueError(f'Probability columns mismatch on {split_name}: lstm={lstm_probs}, gat={gat_probs}')



    merged = lstm.merge(gat, on='row_id', suffixes=('_lstm_meta', '_gat_meta'), how='inner')

    if len(merged) != len(lstm) or len(merged) != len(gat):

        raise ValueError(

            f'Row_id mismatch on {split_name}: merged={len(merged)}, lstm={len(lstm)}, gat={len(gat)}. '

            'Kiem tra hai notebook co dung cung split va row_id khong.'

        )

    if not (merged['true_label_lstm_meta'].astype(int).values == merged['true_label_gat_meta'].astype(int).values).all():

        bad = merged.loc[merged['true_label_lstm_meta'].astype(int) != merged['true_label_gat_meta'].astype(int), 'row_id'].head().tolist()

        raise ValueError(f'True label mismatch on {split_name}; examples: {bad}')



    # Normalize metadata column names after the safety checks.

    merged['true_label'] = merged['true_label_lstm_meta'].astype(int)

    if 'true_label_name_lstm_meta' in merged.columns:

        merged['true_label_name'] = merged['true_label_name_lstm_meta']

    for col in ['split', 'ticker', 'company_name', 'rating_date']:

        a = f'{col}_lstm_meta'

        b = f'{col}_gat_meta'

        if a in merged.columns:

            merged[col] = merged[a]

        elif b in merged.columns:

            merged[col] = merged[b]

    return merged, lstm_probs





val_df, val_prob_cols = align_pair(lstm_val_raw, gat_val_raw, 'val')

test_df, test_prob_cols = align_pair(lstm_test_raw, gat_test_raw, 'test')

if val_prob_cols != test_prob_cols:

    raise ValueError(f'Probability columns mismatch between val and test: val={val_prob_cols}, test={test_prob_cols}')



n_classes = len(val_prob_cols)

class_ids = list(range(n_classes))

if not {'label_id', 'label_name'}.issubset(label_map.columns):

    raise ValueError("label_mapping.csv must contain columns: 'label_id', 'label_name'")

label_ids = sorted(label_map['label_id'].astype(int).tolist())

if label_ids != class_ids:

    raise ValueError(f'label_mapping.csv label_id mismatch: expected {class_ids}, got {label_ids}')

id_to_name = {int(r.label_id): str(r.label_name) for r in label_map.itertuples(index=False)}

class_names = [id_to_name.get(i, str(i)) for i in class_ids]



print('Aligned val/test:', val_df.shape, test_df.shape, '| n_classes:', n_classes)

print('Agreement on test:', float((test_df['lstm_pred_label'] == test_df['gat_pred_label']).mean()))

display(test_df[['row_id', 'true_label', 'lstm_pred_label', 'gat_pred_label']].head())


# ----------------------------------------

def model_prob_matrix(frame, model_key):

    cols = [f'{model_key}_prob_{i}' for i in class_ids]

    missing = [c for c in cols if c not in frame.columns]

    if missing:

        raise ValueError(f'Missing probability columns for {model_key}: {missing}')

    probs = frame[cols].to_numpy(dtype=float)

    probs = np.clip(probs, 1e-9, 1.0)

    return probs / probs.sum(axis=1, keepdims=True)





def compute_metrics(y_true, y_pred, proba=None):

    y_true = np.asarray(y_true, dtype=int)

    y_pred = np.asarray(y_pred, dtype=int)

    out = {

        'Accuracy': float(accuracy_score(y_true, y_pred)),

        'Precision_Weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),

        'Recall_Weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),

        'Macro_F1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),

        'Weighted_F1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),

        'QWK': float(cohen_kappa_score(y_true, y_pred, weights='quadratic')),

        'Ordinal_MAE': float(np.mean(np.abs(y_true - y_pred))),

    }

    if proba is not None:

        try:

            y_bin = label_binarize(y_true, classes=class_ids)

            out['AUC'] = float(roc_auc_score(y_bin, proba, average='macro', multi_class='ovr'))

        except Exception:

            out['AUC'] = float('nan')

    else:

        out['AUC'] = float('nan')

    return out





def prediction_alignment_report(frame, model_key, split_name):

    exported_pred = frame[f'{model_key}_pred_label'].to_numpy(dtype=int)

    prob_pred = model_prob_matrix(frame, model_key).argmax(axis=1)

    mismatch = exported_pred != prob_pred

    return {

        'split': split_name,

        'model': model_key,

        'rows': int(len(frame)),

        'pred_label_argmax_mismatches': int(mismatch.sum()),

        'mismatch_rate': float(mismatch.mean()) if len(frame) else 0.0,

    }





val_lstm_probs = model_prob_matrix(val_df, 'lstm')

val_gat_probs = model_prob_matrix(val_df, 'gat')

test_lstm_probs = model_prob_matrix(test_df, 'lstm')

test_gat_probs = model_prob_matrix(test_df, 'gat')

y_val = val_df['true_label'].to_numpy(dtype=int)

y_test = test_df['true_label'].to_numpy(dtype=int)



# Algorithm 2 defines o_j as the classifier's predicted class label. Use the exported

# pred_label from each base-model artifact instead of recomputing it from probabilities.

val_lstm_pred = val_df['lstm_pred_label'].to_numpy(dtype=int)

val_gat_pred = val_df['gat_pred_label'].to_numpy(dtype=int)

test_lstm_pred = test_df['lstm_pred_label'].to_numpy(dtype=int)

test_gat_pred = test_df['gat_pred_label'].to_numpy(dtype=int)



prediction_alignment = pd.DataFrame([

    prediction_alignment_report(val_df, 'lstm', 'val'),

    prediction_alignment_report(val_df, 'gat', 'val'),

    prediction_alignment_report(test_df, 'lstm', 'test'),

    prediction_alignment_report(test_df, 'gat', 'test'),

])

display(prediction_alignment)



validation_accuracy_weights = {

    'lstm': float(accuracy_score(y_val, val_lstm_pred)),

    'gat': float(accuracy_score(y_val, val_gat_pred)),

}

constant_weights = {'lstm': 1.0, 'gat': 1.0}



# Paper experiments state that a constant weighting factor was assigned to each model.

# Switch to 'validation_accuracy' only for ablation, not for the faithful paper setting.

DCS_MODEL_WEIGHT_MODE = 'constant'  # options: 'constant', 'validation_accuracy'

if DCS_MODEL_WEIGHT_MODE == 'constant':

    dcs_model_weights = constant_weights

elif DCS_MODEL_WEIGHT_MODE == 'validation_accuracy':

    dcs_model_weights = validation_accuracy_weights

else:

    raise ValueError(f'Unsupported DCS_MODEL_WEIGHT_MODE: {DCS_MODEL_WEIGHT_MODE}')



print('Validation accuracy weights:', validation_accuracy_weights)

print('DCS model weights v_j:', dcs_model_weights, '| mode:', DCS_MODEL_WEIGHT_MODE)


# ----------------------------------------

def classifier_correlation_paper(y_true, pred_lstm, pred_gat):

    y_true = np.asarray(y_true, dtype=int)

    pred_lstm = np.asarray(pred_lstm, dtype=int)

    pred_gat = np.asarray(pred_gat, dtype=int)



    lstm_correct = pred_lstm == y_true

    gat_correct = pred_gat == y_true



    n_tt = int(np.sum(lstm_correct & gat_correct))

    n_tf = int(np.sum(lstm_correct & ~gat_correct))

    n_ft = int(np.sum(~lstm_correct & gat_correct))

    n_ff = int(np.sum(~lstm_correct & ~gat_correct))

    denominator = n_ft + n_tf + (2 * n_ff)

    rho = 1.0 if denominator == 0 else float((2 * n_ff) / denominator)



    total = int(len(y_true))

    return {

        'classifier_1': 'LSTM',

        'classifier_2': 'GAT',

        'N_TT_both_correct': n_tt,

        'N_TF_lstm_correct_gat_wrong': n_tf,

        'N_FT_lstm_wrong_gat_correct': n_ft,

        'N_FF_both_wrong': n_ff,

        'rho_eq10': rho,

        'lstm_accuracy': float(np.mean(lstm_correct)),

        'gat_accuracy': float(np.mean(gat_correct)),

        'agreement_rate': float(np.mean(pred_lstm == pred_gat)),

        'disagreement_rate': float(np.mean(pred_lstm != pred_gat)),

        'ideal_dcs_upper_bound_acc': float(np.mean(lstm_correct | gat_correct)),

        'oracle_gain_over_best_single': float(np.mean(lstm_correct | gat_correct) - max(np.mean(lstm_correct), np.mean(gat_correct))),

        'total_rows': total,

    }





classifier_correlation = pd.DataFrame([

    {'split': 'val', **classifier_correlation_paper(y_val, val_lstm_pred, val_gat_pred)},

    {'split': 'test', **classifier_correlation_paper(y_test, test_lstm_pred, test_gat_pred)},

])



display(classifier_correlation)



correlation_path = DMF_ARTIFACT_DIR / 'lstm_gat_classifier_correlation.csv'

classifier_correlation.to_csv(correlation_path, index=False, encoding='utf-8-sig')

print('Saved:', correlation_path)


# ----------------------------------------

DCS_TOP_K = 80  # Paper default for disagreement cases.





def topk_validation_competence(val_probs, val_labels, predicted_class, k=DCS_TOP_K):

    predicted_class = int(predicted_class)

    class_mask = np.asarray(val_labels, dtype=int) == predicted_class

    scores = np.asarray(val_probs[class_mask, predicted_class], dtype=float)

    if len(scores) == 0:

        return 0.0, 0, 0

    k_eff = min(int(k), len(scores))

    if k_eff <= 0:

        return 0.0, 0, int(len(scores))

    top = np.partition(scores, -k_eff)[-k_eff:]

    return float(np.mean(top)), int(k_eff), int(len(scores))





def dcs_fuse_row(i):

    lstm_label = int(test_lstm_pred[i])

    gat_label = int(test_gat_pred[i])

    if lstm_label == gat_label:

        return {

            'final_label': lstm_label,

            'selected_model': 'agreement',

            'dcs_case': 'agreement',

            'lstm_delta': np.nan,

            'gat_delta': np.nan,

            'lstm_k_eff': np.nan,

            'gat_k_eff': np.nan,

            'lstm_reference_count': np.nan,

            'gat_reference_count': np.nan,

            'lstm_weight': np.nan,

            'gat_weight': np.nan,

            'lstm_score': np.nan,

            'gat_score': np.nan,

        }



    lstm_delta, lstm_k_eff, lstm_reference_count = topk_validation_competence(

        val_lstm_probs, y_val, lstm_label, DCS_TOP_K

    )

    gat_delta, gat_k_eff, gat_reference_count = topk_validation_competence(

        val_gat_probs, y_val, gat_label, DCS_TOP_K

    )

    lstm_weight = float(dcs_model_weights['lstm'])

    gat_weight = float(dcs_model_weights['gat'])

    lstm_test_score = float(test_lstm_probs[i, lstm_label])

    gat_test_score = float(test_gat_probs[i, gat_label])

    lstm_score = (lstm_delta * lstm_weight) + lstm_test_score

    gat_score = (gat_delta * gat_weight) + gat_test_score



    if lstm_score >= gat_score:

        final_label = lstm_label

        selected_model = 'lstm'

    else:

        final_label = gat_label

        selected_model = 'gat'



    return {

        'final_label': final_label,

        'selected_model': selected_model,

        'dcs_case': 'disagreement',

        'lstm_delta': lstm_delta,

        'gat_delta': gat_delta,

        'lstm_k_eff': lstm_k_eff,

        'gat_k_eff': gat_k_eff,

        'lstm_reference_count': lstm_reference_count,

        'gat_reference_count': gat_reference_count,

        'lstm_weight': lstm_weight,

        'gat_weight': gat_weight,

        'lstm_score': lstm_score,

        'gat_score': gat_score,

    }





dcs_rows = [dcs_fuse_row(i) for i in range(len(test_df))]

dcs_df = pd.DataFrame(dcs_rows)

dmf_pred = dcs_df['final_label'].to_numpy(dtype=int)



# The selected classifier gives the final probability vector in DCS; agreement uses average probs for reporting AUC.

dmf_probs = np.zeros_like(test_lstm_probs)

for i, selected in enumerate(dcs_df['selected_model']):

    if selected == 'lstm':

        dmf_probs[i] = test_lstm_probs[i]

    elif selected == 'gat':

        dmf_probs[i] = test_gat_probs[i]

    else:

        dmf_probs[i] = (test_lstm_probs[i] + test_gat_probs[i]) / 2.0



dcs_summary = dcs_df['dcs_case'].value_counts().rename_axis('case').reset_index(name='count')

dcs_summary['ratio'] = dcs_summary['count'] / len(dcs_df)

display(dcs_summary)

print('DCS_TOP_K:', DCS_TOP_K)

print('DCS_MODEL_WEIGHT_MODE:', DCS_MODEL_WEIGHT_MODE)


# ----------------------------------------

soft_probs = (test_lstm_probs + test_gat_probs) / 2.0

soft_pred = soft_probs.argmax(axis=1)



weight_sum = float(validation_accuracy_weights['lstm'] + validation_accuracy_weights['gat'])

weighted_probs = (

    validation_accuracy_weights['lstm'] * test_lstm_probs

    + validation_accuracy_weights['gat'] * test_gat_probs

) / max(weight_sum, 1e-12)

weighted_pred = weighted_probs.argmax(axis=1)



results = pd.DataFrame([

    {'Model': 'LSTM only', **compute_metrics(y_test, test_lstm_pred, test_lstm_probs)},

    {'Model': 'GAT only', **compute_metrics(y_test, test_gat_pred, test_gat_probs)},

    {'Model': 'Soft voting', **compute_metrics(y_test, soft_pred, soft_probs)},

    {'Model': 'Validation-weighted voting', **compute_metrics(y_test, weighted_pred, weighted_probs)},

    {'Model': f'DMF/DCS proposed ({DCS_MODEL_WEIGHT_MODE})', **compute_metrics(y_test, dmf_pred, dmf_probs)},

])



display(results)

metrics_path = DMF_ARTIFACT_DIR / 'dmf_dcs_metrics.csv'

results.to_csv(metrics_path, index=False, encoding='utf-8-sig')

print('Saved:', metrics_path)


# ----------------------------------------

disagreement_mask = test_lstm_pred != test_gat_pred

if np.any(disagreement_mask):

    disagreement_metrics = pd.DataFrame([

        {'Model': 'LSTM on disagreements', **compute_metrics(y_test[disagreement_mask], test_lstm_pred[disagreement_mask], test_lstm_probs[disagreement_mask])},

        {'Model': 'GAT on disagreements', **compute_metrics(y_test[disagreement_mask], test_gat_pred[disagreement_mask], test_gat_probs[disagreement_mask])},

        {'Model': f'DMF/DCS on disagreements ({DCS_MODEL_WEIGHT_MODE})', **compute_metrics(y_test[disagreement_mask], dmf_pred[disagreement_mask], dmf_probs[disagreement_mask])},

    ])

    display(disagreement_metrics)

    display(dcs_df.loc[disagreement_mask, 'selected_model'].value_counts().rename_axis('selected_model').reset_index(name='count'))

    display(dcs_df.loc[disagreement_mask, [

        'selected_model', 'lstm_delta', 'gat_delta', 'lstm_k_eff', 'gat_k_eff',

        'lstm_reference_count', 'gat_reference_count', 'lstm_weight', 'gat_weight',

        'lstm_score', 'gat_score',

    ]].describe(include='all'))

else:

    print('Khong co mau bat dong giua LSTM va GAT tren test set.')


# ----------------------------------------

out = test_df[['row_id', 'split', 'ticker', 'company_name', 'rating_date', 'true_label']].copy()

out['true_label_name'] = [id_to_name.get(int(y), str(y)) for y in out['true_label']]

out['lstm_pred_label'] = test_lstm_pred

out['lstm_pred_label_name'] = [id_to_name.get(int(y), str(y)) for y in test_lstm_pred]

out['lstm_confidence'] = test_lstm_probs[np.arange(len(test_lstm_pred)), test_lstm_pred]

out['gat_pred_label'] = test_gat_pred

out['gat_pred_label_name'] = [id_to_name.get(int(y), str(y)) for y in test_gat_pred]

out['gat_confidence'] = test_gat_probs[np.arange(len(test_gat_pred)), test_gat_pred]

out['dmf_pred_label'] = dmf_pred

out['dmf_pred_label_name'] = [id_to_name.get(int(y), str(y)) for y in dmf_pred]

out['dmf_confidence'] = dmf_probs[np.arange(len(dmf_pred)), dmf_pred]

out = pd.concat([out.reset_index(drop=True), dcs_df.reset_index(drop=True)], axis=1)

for cls_idx in class_ids:

    out[f'lstm_prob_{cls_idx}'] = test_lstm_probs[:, cls_idx]

    out[f'gat_prob_{cls_idx}'] = test_gat_probs[:, cls_idx]

    out[f'dmf_prob_{cls_idx}'] = dmf_probs[:, cls_idx]



prediction_path = DMF_ARTIFACT_DIR / 'dmf_dcs_test_predictions.csv'

out.to_csv(prediction_path, index=False, encoding='utf-8-sig')

print('Saved:', prediction_path)

display(out.head())


# ----------------------------------------

cm = confusion_matrix(y_test, dmf_pred, labels=class_ids)

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)



plt.figure(figsize=(6, 5), dpi=160)

sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.title('DMF/DCS Confusion Matrix (Test)')

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.tight_layout()

cm_plot_path = DMF_ARTIFACT_DIR / 'dmf_dcs_test_confusion_matrix.png'

plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')

plt.show()



display(cm_df)

print('Classification report (DMF/DCS test set):')

print(classification_report(

    y_test,

    dmf_pred,

    labels=class_ids,

    target_names=class_names,

    digits=4,

    zero_division=0,

))



cls_report_df = pd.DataFrame(

    classification_report(

        y_test,

        dmf_pred,

        labels=class_ids,

        target_names=class_names,

        output_dict=True,

        zero_division=0,

    )

).transpose()

cm_csv_path = DMF_ARTIFACT_DIR / 'dmf_dcs_test_confusion_matrix.csv'

cls_csv_path = DMF_ARTIFACT_DIR / 'dmf_dcs_test_classification_report.csv'

cm_df.to_csv(cm_csv_path, encoding='utf-8-sig')

cls_report_df.to_csv(cls_csv_path, encoding='utf-8-sig')

print('Saved:', cm_plot_path)

print('Saved:', cm_csv_path)

print('Saved:', cls_csv_path)


# ----------------------------------------

# Inspect disagreement cases where DCS actually selected between LSTM and GAT.

disagreements = out[out['dcs_case'].eq('disagreement')].copy()

if len(disagreements):

    preview_cols = [

        'row_id', 'ticker', 'rating_date', 'true_label_name',

        'lstm_pred_label_name', 'gat_pred_label_name', 'dmf_pred_label_name',

        'selected_model', 'lstm_score', 'gat_score',

    ]

    display(disagreements[preview_cols].head(20))

else:

    print('Khong co case bat dong giua LSTM va GAT tren test set.')


# ----------------------------------------
