"""Reconcile held-out predictions and training/test separation independently."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, f1_score, confusion_matrix
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'artifacts/holdout/adult'
OUT = ROOT/'codes/verification/research/adult_holdout'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    threadpool_limits(1)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((DATA/'protocol.json').read_text())
    spec = protocol['specification']
    complete = json.loads((DATA/'complete.json').read_text())
    assert complete['protocol_sha256'] == sha(DATA/'protocol.json')
    for p, checksum in spec['code_sha256'].items():
        assert sha(ROOT/p) == checksum
    for p, checksum in complete['output_sha256'].items():
        assert sha(DATA/p) == checksum
    raw, entry = Catalog(ROOT/'data').load('adult')
    assert entry['sha256'] == spec['source_sha256']
    split = pd.read_csv(DATA/'split.csv').set_index('row_id')
    assert sha(DATA/'split.csv') == spec['split_sha256']
    train = raw.loc[raw._source_split == 'train'].copy()
    test_raw = raw.loc[raw._source_split == 'original_test'].copy()
    # Direct column joins, independent of the hash-based exclusion code.
    merged = test_raw.merge(train[spec['fingerprint_columns']].drop_duplicates(), on=spec['fingerprint_columns'], how='left', indicator=True, validate='many_to_one')
    excluded = set(merged.loc[merged['_merge'] == 'both', '_row_id'])
    included = set(merged.loc[merged['_merge'] == 'left_only', '_row_id'])
    assert len(excluded) == 26 and len(included) == 16255
    assert set(split.query("partition == 'excluded'").index) == excluded
    assert set(split.query("partition == 'test'").index) == included
    assert set(split.query("partition == 'train'").index) == set(train._row_id)
    assert not set(split.query("partition == 'train'").fingerprint) & set(split.query("partition == 'test'").fingerprint)
    inner = json.loads((DATA/'inner_splits.json').read_text())
    validation_ids = []
    for fold in inner:
        a, b = set(fold['train_row_ids']), set(fold['validation_row_ids'])
        assert not a & b and a | b == set(train._row_id)
        assert not set(split.loc[list(a), 'fingerprint']) & set(split.loc[list(b), 'fingerprint'])
        validation_ids += list(b)
    assert len(validation_ids) == len(set(validation_ids)) == len(train)
    locked = json.loads((DATA/'models_locked.json').read_text())
    evaluation = json.loads((DATA/'evaluation_started.json').read_text())
    assert locked['locked_at_utc'] >= protocol['frozen_at_utc']
    assert evaluation['started_at_utc'] >= locked['locked_at_utc']
    predictions = pd.read_csv(DATA/'predictions.csv')
    assert set(predictions.row_id) == included and predictions.row_id.is_unique
    actual = raw.set_index('_row_id').loc[predictions.row_id, 'income_gt_50k'].to_numpy()
    assert np.array_equal(actual, predictions.actual)
    assert np.array_equal(predictions.fingerprint, split.loc[predictions.row_id, 'fingerprint'])
    metrics = pd.read_csv(DATA/'metrics.csv').set_index('model')
    checks, max_error = [], 0.
    for name in spec['models']:
        assert sha(DATA/f'{name}.joblib') == locked['model_sha256'][name]
        marker = json.loads((DATA/f'{name}_fit.json').read_text())
        assert protocol['frozen_at_utc'] <= marker['finished_at_utc'] <= locked['locked_at_utc']
        model = joblib.load(DATA/f'{name}.joblib')
        inputs = raw.set_index('_row_id').loc[predictions.row_id, spec['features']]
        probabilities = model.predict_proba(inputs)[:, 1]
        assert np.allclose(probabilities, predictions[name], rtol=0, atol=1e-14)
        if name == 'Mean':
            assert np.allclose(probabilities, train.income_gt_50k.mean(), rtol=0, atol=1e-14)
        pre = model.named_steps['pre']
        numeric = list(train[spec['features']].select_dtypes(include='number').columns)
        assert np.allclose(pre.named_transformers_['num'].named_steps['impute'].statistics_, train[numeric].median())
        p = predictions[name].to_numpy()
        values = {'auc': roc_auc_score(actual, p), 'ap': average_precision_score(actual, p),
                  'brier': brier_score_loss(actual, p), 'log_loss': log_loss(actual, p), 'f1': f1_score(actual, p >= .5)}
        for metric_name, value in values.items():
            error = abs(value - metrics.loc[name, metric_name])
            max_error = max(max_error, error)
            assert error < 1e-12
            checks.append({'model': name, 'metric': metric_name, 'saved': metrics.loc[name, metric_name], 'recomputed': value, 'absolute_error': error})
    bootstrap = np.load(DATA/'bootstrap_auc.npz')
    assert list(bootstrap['models']) == spec['models']
    samples = bootstrap['auc']
    groups, group_codes = np.unique(predictions.fingerprint, return_inverse=True)
    rng = np.random.default_rng(spec['uncertainty']['seed'])
    # Recreate selected resamples and use sklearn's independent weighted AUC.
    selected_replicates = {0, 1, 2, 499, 999, 1499, 1999}
    for b in range(2000):
        counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        if b in selected_replicates:
            for j, name in enumerate(spec['models']):
                value = roc_auc_score(actual, predictions[name], sample_weight=counts[group_codes])
                assert abs(value - samples[b, j]) < 1e-12
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    assert np.allclose(lo, metrics.loc[spec['models'], 'auc_ci_low'], rtol=0, atol=1e-12)
    assert np.allclose(hi, metrics.loc[spec['models'], 'auc_ci_high'], rtol=0, atol=1e-12)
    pairs = pd.read_csv(DATA/'paired_differences.csv').set_index('comparator')
    j = spec['models'].index(spec['selected_before_test'])
    for k, name in enumerate(spec['models']):
        if k == j:
            continue
        bounds = np.percentile(samples[:, j] - samples[:, k], [2.5, 97.5])
        assert np.allclose(bounds, pairs.loc[name, ['ci_low', 'ci_high']].to_numpy(dtype=float), rtol=0, atol=1e-12)
        assert abs(pairs.loc[name, 'auc_difference'] - (metrics.loc[spec['selected_before_test'], 'auc'] - metrics.loc[name, 'auc'])) < 1e-12
    trials = 0
    for name in spec['models']:
        if '_' not in name:
            continue
        trace = json.loads((DATA/f'{name}_search.json').read_text())
        assert len(trace['rows']) == 12
        assert trace['best_params'] == json.loads((DATA/f'{name}_fit.json').read_text())['selected_params']
        trials += len(trace['rows']) * len(inner)
    assert trials == 144
    pd.DataFrame(checks).to_csv(OUT/'metric_reconciliation.csv', index=False)
    report = {'status': 'passed', 'metrics_recomputed': len(checks), 'max_metric_absolute_error': max_error,
              'inner_folds_checked': len(inner), 'training_test_fingerprint_overlap': 0,
              'bootstrap_sklearn_comparisons': len(selected_replicates)*len(spec['models']),
              'primary_confusion_matrix_tn_fp_fn_tp': confusion_matrix(actual, predictions[spec['selected_before_test']] >= .5).ravel().tolist(),
              'training_rows': len(train), 'test_rows': len(included), 'excluded_rows': len(excluded),
              'test_groups': len(groups), 'inner_fits': trials, 'final_refits': len(spec['models']),
              'protocol_sha256': sha(DATA/'protocol.json'), 'source_hash_unchanged': True}
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
