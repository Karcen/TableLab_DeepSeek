"""One reserved-test evaluation, after freezing all training-only decisions."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog, SPECS
from tablelab.engine import predictor_fingerprints
from tablelab.research.experiment import pipeline, splits, search, prediction, metric, dump, space

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/holdout/adult'
MODELS = ['Mean', 'Linear', 'Ridge', 'Sparse', 'Tree', 'RF', 'ExtraTrees', 'HGB', 'KNN', 'RF_grid', 'RF_tpe', 'HGB_grid', 'HGB_tpe']
PRIMARY = 'HGB_grid'
BOOTSTRAPS = 2000
BOOT_SEED = 20261102


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepared_auc(y, p):
    order = np.argsort(p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    yy = np.asarray(y)[order]
    def auc(weights):
        w = weights[order]
        pos = np.add.reduceat(w * yy, starts)
        neg = np.add.reduceat(w * (1 - yy), starts)
        return np.dot(pos, np.cumsum(neg) - .5 * neg) / (pos.sum() * neg.sum())
    return auc


def main():
    threadpool_limits(1)
    OUT.mkdir(parents=True, exist_ok=True)
    df, entry = Catalog(ROOT / 'data').load('adult')
    cols = [c for c in df if not c.startswith('_') and c not in ('income', 'income_gt_50k')]
    assert len(cols) == 14
    fp = predictor_fingerprints(df[cols])
    train = df.loc[df._source_split == 'train'].reset_index(drop=True)
    excluded = (df._source_split == 'original_test') & fp.isin(set(fp[df._source_split == 'train']))
    eligible = (df._source_split == 'original_test') & ~excluded
    test = df.loc[eligible].reset_index(drop=True)
    test_fp = fp.loc[eligible].reset_index(drop=True)
    assert (len(train), len(test), int(excluded.sum()), test_fp.nunique()) == (32561, 16255, 26, 16249)
    audit = ROOT / 'codes/verification/research/independence_audit/adult_test_overlap_flags.csv'
    flags = pd.read_csv(audit)
    assert set(flags.loc[flags.candidate_after_cross_file_exclusion, 'row_id']) == set(test._row_id)
    features = list(SPECS['adult'].defaults)
    cv = splits(train, 'adult', 3)
    split_frame = pd.DataFrame({'row_id': df._row_id, 'partition': 'excluded', 'fingerprint': fp})
    split_frame.loc[df._source_split == 'train', 'partition'] = 'train'
    split_frame.loc[eligible, 'partition'] = 'test'
    split_bytes = split_frame.to_csv(index=False).encode('utf-8')
    code_paths = [Path(__file__), ROOT/'codes/tablelab/research/experiment.py', ROOT/'codes/tablelab/engine.py', ROOT/'codes/tablelab/catalog.py']
    specification = {
        'dataset': 'adult', 'source_sha256': entry['sha256'],
        'audit_flags_sha256': sha(audit), 'prior_cv_summary_sha256': sha(ROOT/'artifacts/research/summary.csv'),
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in code_paths},
        'split_sha256': hashlib.sha256(split_bytes).hexdigest(),
        'training_rows': len(train), 'test_rows': len(test), 'excluded_cross_file_rows': int(excluded.sum()),
        'test_fingerprint_groups': int(test_fp.nunique()), 'fingerprint_columns': cols,
        'features': features, 'target': 'income_gt_50k', 'models': MODELS, 'selected_before_test': PRIMARY,
        'primary_metric': 'auc', 'secondary_metrics': ['ap', 'brier', 'log_loss', 'f1'], 'threshold': .5,
        'inner_cv': '3-fold StratifiedGroupKFold on 14-predictor fingerprints, shuffle=True, random_state=42',
        'search': {'trials': 12, 'tpe_seed': 42, 'tpe_startup_trials': 4, 'domains': {k: space(k) for k in ('RF', 'HGB')}},
        'fit_count': {'inner': 144, 'final': 13},
        'test_duplicates': 'Retain all rows; cluster bootstrap by full-predictor fingerprint',
        'uncertainty': {'replicates': BOOTSTRAPS, 'seed': BOOT_SEED, 'method': 'paired nonparametric cluster bootstrap, percentile 95% CI',
                        'comparisons': 'HGB_grid minus each other flow; descriptive intervals, no multiplicity adjustment',
                        'scope': 'Conditional on fitted models; test sampling uncertainty, not training or tuning variance'},
        'test_access_history': 'Previously profiled for content and overlap; not previously used for fitting or saved performance estimates',
        'evaluation_rule': 'Freeze protocol before fitting; lock all 13 fitted models before predictions; no test-driven refitting, model selection, or threshold selection',
        'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'optuna', 'joblib')},
    }
    protocol_file = OUT/'protocol.json'
    if protocol_file.exists():
        protocol = json.loads(protocol_file.read_text())
        assert protocol['specification'] == specification, 'Frozen protocol differs; do not overwrite evaluated study'
    else:
        dump(protocol_file, {'frozen_at_utc': utc(), 'specification': specification})
        (OUT/'split.csv').write_bytes(split_bytes)
        dump(OUT/'inner_splits.json', [{'train_row_ids': train.iloc[a]._row_id.tolist(), 'validation_row_ids': train.iloc[b]._row_id.tolist()} for a, b in cv])
    protocol_hash = sha(protocol_file)
    if (OUT/'complete.json').exists():
        assert json.loads((OUT/'complete.json').read_text())['protocol_sha256'] == protocol_hash
        print('Existing held-out evaluation retained; no refit.', flush=True)
        return
    X = train[features]
    y = train.income_gt_50k
    for name in MODELS:
        model_path = OUT/f'{name}.joblib'
        marker = OUT/f'{name}_fit.json'
        if marker.exists():
            cached = json.loads(marker.read_text())
            assert cached['protocol_sha256'] == protocol_hash and cached['model_sha256'] == sha(model_path)
            continue
        assert not (OUT/'models_locked.json').exists(), 'Models already locked'
        start = time.perf_counter()
        print(f'Fitting {name} on original train only', flush=True)
        if '_' in name:
            family, method = name.split('_')
            fit, params = search(pipeline(X, family, True), X, y, cv, True, method, 0, OUT/f'{name}_search.json')
        else:
            fit = pipeline(X, name, True).fit(X, y)
            params = {}
        joblib.dump(fit, model_path, compress=3)
        dump(marker, {'protocol_sha256': protocol_hash, 'model_sha256': sha(model_path), 'finished_at_utc': utc(),
                      'seconds': time.perf_counter()-start, 'selected_params': params,
                      'estimator_params': fit.named_steps['model'].get_params()})
        print(f'Finished {name} in {time.perf_counter()-start:.1f}s', flush=True)
    model_hashes = {name: sha(OUT/f'{name}.joblib') for name in MODELS}
    lock = OUT/'models_locked.json'
    if lock.exists():
        assert json.loads(lock.read_text())['model_sha256'] == model_hashes
    else:
        dump(lock, {'locked_at_utc': utc(), 'protocol_sha256': protocol_hash, 'model_sha256': model_hashes,
                    'test_predictions_computed': False})
    # No test predictions or metrics are obtained above this point.
    dump(OUT/'evaluation_started.json', {'started_at_utc': utc(), 'models_lock_sha256': sha(lock)})
    yy = test.income_gt_50k.to_numpy()
    predictions = pd.DataFrame({'row_id': test._row_id, 'fingerprint': test_fp, 'actual': yy})
    rows = []
    for name in MODELS:
        fit = joblib.load(OUT/f'{name}.joblib')
        pp = prediction(fit, test[features], True)
        predictions[name] = pp
        rows.append({'model': name, 'n': len(yy), **metric(yy, pp, True)})
    predictions.to_csv(OUT/'predictions.csv', index=False)
    scores = pd.DataFrame(rows).set_index('model')
    funcs = [prepared_auc(yy, predictions[name].to_numpy()) for name in MODELS]
    _, group_codes = np.unique(test_fp, return_inverse=True)
    groups = int(group_codes.max()+1)
    rng = np.random.default_rng(BOOT_SEED)
    samples = np.empty((BOOTSTRAPS, len(MODELS)))
    for b in range(BOOTSTRAPS):
        counts = np.bincount(rng.integers(groups, size=groups), minlength=groups)
        weights = counts[group_codes]
        for j, f in enumerate(funcs):
            samples[b, j] = f(weights)
            if b < 3:
                assert np.isclose(samples[b, j], roc_auc_score(yy, predictions[MODELS[j]], sample_weight=weights), rtol=0, atol=1e-12)
    for j, f in enumerate(funcs):
        assert abs(f(np.ones(len(yy))) - scores.loc[MODELS[j], 'auc']) < 1e-12
    np.savez_compressed(OUT/'bootstrap_auc.npz', auc=samples, models=np.array(MODELS))
    scores['auc_ci_low'], scores['auc_ci_high'] = np.quantile(samples, [.025, .975], axis=0)
    scores['test_rank'] = scores.auc.rank(method='min', ascending=False).astype(int)
    previous = pd.read_csv(ROOT/'artifacts/research/summary.csv').query("dataset == 'adult'").set_index('model')
    scores['cv_auc'] = previous['mean']
    scores['cv_rank'] = previous['mean'].rank(method='min', ascending=False).astype(int)
    scores.to_csv(OUT/'metrics.csv')
    primary_index = MODELS.index(PRIMARY)
    comparisons = []
    for j, name in enumerate(MODELS):
        if name == PRIMARY:
            continue
        lo, hi = np.quantile(samples[:, primary_index] - samples[:, j], [.025, .975])
        comparisons.append({'primary': PRIMARY, 'comparator': name,
                            'auc_difference': scores.loc[PRIMARY, 'auc'] - scores.loc[name, 'auc'], 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(comparisons).to_csv(OUT/'paired_differences.csv', index=False)
    dump(OUT/'complete.json', {'completed_at_utc': utc(), 'protocol_sha256': protocol_hash, 'models_lock_sha256': sha(lock),
                              'test_rows': len(yy), 'test_positive': int(yy.sum()), 'bootstrap_replicates': BOOTSTRAPS,
                              'inner_fits': 144, 'final_refits': 13,
                              'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix != '.joblib' and p.name != 'complete.json'}})
    print(scores.sort_values('auc', ascending=False).to_string(), flush=True)


if __name__ == '__main__':
    main()
