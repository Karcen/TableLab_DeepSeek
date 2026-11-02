"""Expanded, separately versioned nested benchmark; original studies stay intact."""
from pathlib import Path
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from tablelab.catalog import Catalog, SPECS
from tablelab.research.experiment import load_data, splits, preprocess, metric, dump, ABLAT

MODELS = ['Mean', 'Linear', 'Ridge', 'Sparse', 'Tree', 'RF', 'ExtraTrees', 'HGB', 'KNN', 'AdaBoost', 'SVM', 'XGB', 'LightGBM', 'CatBoost', 'MLP', 'ResMLP', 'DCN', 'GatedMLP', 'FeatureTransformer', 'ELM', 'RBFNet']
NEURAL = ['MLP', 'ResMLP', 'DCN', 'GatedMLP', 'FeatureTransformer']
RECURRENT = ['RNN', 'GRU', 'BiGRU', 'LSTM', 'BiLSTM', 'TCN', 'SeqTransformer']
SEQUENCE_MODELS = ['Mean', 'Persistence', 'Linear', 'Ridge', 'RF', 'XGB', 'LightGBM', 'MLP'] + RECURRENT
SEQ_FIELDS = ['age', 'player_height', 'player_weight', 'pts', 'reb', 'ast', 'gp']
SEQ_ABLATIONS = {'No age': ['age'], 'No physique': ['player_height','player_weight'],
                 'No past production': ['pts','reb','ast'], 'No availability': ['gp'], 'No oldest season': []}
ARCH_ABLATIONS = {'MLP':'No activations','ResMLP':'No residuals','DCN':'No cross branch','GatedMLP':'No gates','FeatureTransformer':'No attention'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def domains(model, cls=False):
    if model in ['Mean', 'Persistence']:
        return {}
    if model == 'Linear':
        return {'fit_intercept': [True,False]}
    if model == 'Ridge':
        return {'alpha': [.001,.01,.1,1.,10.,100.,1000.,10000.]}
    if model == 'Sparse':
        return {'alpha': [.01,.1,1.,10.], 'l1_ratio': [.5,1.]}
    if model == 'Tree':
        return {'max_depth': [2,4,8,None], 'min_samples_leaf': [2,10]}
    if model in ['RF', 'ExtraTrees']:
        return {'max_depth': [6,12], 'min_samples_leaf': [3,10], 'max_features': [.7,1.]}
    if model == 'HGB':
        return {'max_leaf_nodes': [7,31], 'learning_rate': [.05,.1], 'l2_regularization': [0.,1.]}
    if model == 'KNN':
        return {'n_neighbors': [3,7,15,31], 'weights': ['uniform','distance']}
    if model == 'AdaBoost':
        return {'max_depth': [1,3], 'learning_rate': [.03,.1], 'n_estimators': [50,100]}
    if model == 'SVM':
        return {'C': [.001,.01,.1,.3,1.,3.,10.,100.]}
    if model == 'XGB':
        return {'max_depth': [3,6], 'learning_rate': [.03,.1], 'reg_lambda': [1.,10.]}
    if model == 'LightGBM':
        return {'num_leaves': [7,31], 'learning_rate': [.03,.1], 'min_child_samples': [10,30]}
    if model == 'CatBoost':
        return {'depth': [4,6], 'learning_rate': [.03,.1], 'l2_leaf_reg': [1.,10.]}
    if model in NEURAL + RECURRENT:
        return {'width': [16,32] if model in ['FeatureTransformer','SeqTransformer'] else [32,64], 'lr': [.003,.01], 'weight_decay': [.0001,.01]}
    if model == 'ELM':
        return {'width': [32,128], 'activation': ['tanh','relu'], 'alpha': [.1,10.]}
    if model == 'RBFNet':
        return {'centres': [16,64], 'gamma': [.1,1.], 'alpha': [.1,10.]}
    raise ValueError(model)


def make_model(name, params, cls, seed, groups, sequence_channels=0, steps=3, epochs=40, calibration_cv=None, ablation=None):
    if name in NEURAL + RECURRENT:
        from tablelab.research.networks import TorchModel
        return TorchModel(name,params,groups,cls,seed,sequence_channels if name in RECURRENT else 0,steps,epochs,ablation)
    if name in ['ELM','RBFNet']:
        from tablelab.research.networks import BasisNetwork
        return BasisNetwork(name,params,cls,seed)
    if name == 'Linear':
        return LogisticRegression(C=np.inf,max_iter=3000,random_state=seed,**params) if cls else LinearRegression(**params)
    if name == 'Ridge':
        return LogisticRegression(C=1/params['alpha'],max_iter=3000,random_state=seed) if cls else Ridge(**params)
    if name == 'Sparse':
        return LogisticRegression(C=1/params['alpha'],l1_ratio=params['l1_ratio'],solver='saga',max_iter=3000,random_state=seed) if cls else ElasticNet(**params,max_iter=10000,random_state=seed)
    if name == 'Tree':
        return (DecisionTreeClassifier if cls else DecisionTreeRegressor)(**params,random_state=seed)
    if name in ['RF','ExtraTrees']:
        t = (RandomForestClassifier if cls else RandomForestRegressor) if name == 'RF' else (ExtraTreesClassifier if cls else ExtraTreesRegressor)
        return t(**params,n_estimators=64,n_jobs=1,random_state=seed)
    if name == 'HGB':
        return (HistGradientBoostingClassifier if cls else HistGradientBoostingRegressor)(**params,max_iter=100,early_stopping=False,random_state=seed)
    if name == 'KNN':
        return (KNeighborsClassifier if cls else KNeighborsRegressor)(**params,n_jobs=1)
    if name == 'AdaBoost':
        p = dict(params)
        tree = (DecisionTreeClassifier if cls else DecisionTreeRegressor)(max_depth=p.pop('max_depth'),random_state=seed)
        return (AdaBoostClassifier if cls else AdaBoostRegressor)(estimator=tree,**p,random_state=seed)
    if name == 'SVM':
        return CalibratedClassifierCV(LinearSVC(**params,random_state=seed,max_iter=5000),cv=calibration_cv if calibration_cv is not None else 3,n_jobs=1) if cls else LinearSVR(**params,epsilon=.1,loss='squared_epsilon_insensitive',dual=False,max_iter=5000,random_state=seed)
    if name == 'XGB':
        from xgboost import XGBRegressor, XGBClassifier
        return (XGBClassifier if cls else XGBRegressor)(**params,n_estimators=120,tree_method='hist',max_bin=63,n_jobs=1,random_state=seed)
    if name == 'LightGBM':
        from lightgbm import LGBMRegressor, LGBMClassifier
        return (LGBMClassifier if cls else LGBMRegressor)(**params,n_estimators=120,reg_lambda=1.,n_jobs=1,verbosity=-1,random_state=seed,deterministic=True,force_col_wise=True)
    if name == 'CatBoost':
        from catboost import CatBoostRegressor, CatBoostClassifier
        return (CatBoostClassifier if cls else CatBoostRegressor)(**params,iterations=120,thread_count=1,random_seed=seed,verbose=False,allow_writing_files=False)
    raise ValueError(name)


def build_sequence(root, out):
    raw, entry = Catalog(Path(root)/'data').load('nba')
    raw = raw.copy()
    raw['year'] = raw.season.str[:4].astype(int)
    key_cols = ['player_name','country','draft_year']
    key = raw[key_cols].astype(str).agg('|'.join,axis=1)
    raw['_person_key'] = key.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    ambiguous = raw.duplicated(['_person_key','year'],keep=False)
    clean = raw.loc[~ambiguous].copy()
    indexed = {(r['_person_key'],int(r['year'])):r for r in clean.to_dict('records')}
    records = []
    missing, inconsistent = 0,0
    for current in clean.to_dict('records'):
        history = [indexed.get((current['_person_key'],current['year']-lag)) for lag in [3,2,1]]
        if any(r is None for r in history):
            missing += 1
            continue
        chain = history+[current]
        if any(not (0 <= b['age']-a['age'] <= 2 and abs(b['player_height']-a['player_height']) <= 5) for a,b in zip(chain,chain[1:])):
            inconsistent += 1
            continue
        row = {'_row_id':current['_row_id'],'season':current['season'],'_person_key':current['_person_key'],'pts':current['pts']}
        for lag,r in zip([3,2,1],history):
            row[f'_history_id_lag{lag}'] = r['_row_id']
            for field in SEQ_FIELDS:
                row[f'lag{lag}_{field}'] = r[field]
        records.append(row)
    frame = pd.DataFrame(records)
    assert frame._row_id.is_unique and frame.notna().all().all()
    assert len(frame)+int(ambiguous.sum())+missing+inconsistent == len(raw)
    features = [f'lag{lag}_{field}' for lag in [3,2,1] for field in SEQ_FIELDS]
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'sequence_master.csv',index=False)
    dump(out/'sequence_audit.json',{'source_sha256':entry['sha256'],'raw_rows':len(raw),'ambiguous_key_season_rows_excluded':int(ambiguous.sum()),
                                   'missing_three_consecutive_prior_seasons':missing,'age_height_continuity_excluded':inconsistent,
                                   'eligible_windows':len(frame),'features':features,'target':'pts',
                                   'key':'hashed player_name + country + draft_year; not a verified player identifier',
                                   'prediction':'one-season-ahead rolling origin with frozen model per validation block; actual prior seasons available at each prediction time',
                                   'identity_limit':'Same-name people may remain indistinguishable; this task has no new independent test cohort'})
    return frame,features


def seq_splits(df):
    seasons = np.array(sorted(df.season.unique()))
    return [(np.flatnonzero(df.season.isin(seasons[a])),np.flatnonzero(df.season.isin(seasons[b]))) for a,b in TimeSeriesSplit(n_splits=3).split(seasons)]


def prepare(train, validation, features, target, cls):
    pre = preprocess(train[features])
    X = pre.fit_transform(train[features]).astype(np.float32)
    V = pre.transform(validation[features]).astype(np.float32)
    names = list(pre.get_feature_names_out())
    groups = [tuple(j for j,n in enumerate(names) if n == 'num__'+col or n.startswith('cat__'+col+'_')) for col in features]
    groups = tuple(g for g in groups if g)
    assert sorted(j for g in groups for j in g) == list(range(len(names)))
    y = train[target].to_numpy()
    centre, scale = (0.,1.) if cls else (float(y.mean()),float(y.std()) or 1.)
    return {'X':X,'y':(y-centre)/scale,'V':V,'actual':validation[target].to_numpy(),
            'centre':centre,'scale':scale,'pre':pre,'groups':groups,'feature_names':features,
            'train_ids':train._row_id.tolist(),'validation_ids':validation._row_id.tolist(),
            'calibration_cv':splits(train.reset_index(drop=True),'adult',3) if cls else None}


def model_predict(model, X, cls, name):
    if name in NEURAL+RECURRENT+['ELM','RBFNet']:
        return model.predict(X)
    return model.predict_proba(X)[:,1] if cls else model.predict(X)


def get_metric(actual, p, cls):
    return metric(actual,p,cls)


def objective_score(actual,p,cls):
    from sklearn.metrics import roc_auc_score, mean_squared_error
    return float(roc_auc_score(actual,p)) if cls else -float(np.sqrt(mean_squared_error(actual,p)))


def search_model(name,method,cache,cls,seed,channels,steps,path,ablation=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    grid = list(ParameterGrid(domains(name,cls)))
    rows, seen = [],{}
    def evaluate(pars, trial):
        code = json.dumps(pars,sort_keys=True)
        start = time.perf_counter()
        reused = code in seen
        if not reused:
            values = []
            for k,fold in enumerate(cache['inner']):
                fit = make_model(name,pars,cls,seed+k+1,fold['groups'],channels,steps,calibration_cv=fold['calibration_cv'],ablation=ablation)
                fit.fit(fold['X'],fold['y'])
                pp = model_predict(fit,fold['V'],cls,name)*fold['scale']+fold['centre']
                values.append(objective_score(fold['actual'],pp,cls))
            seen[code] = values
        values = seen[code]
        rows.append({'trial':trial,'params':pars,'fold_scores':values,'score':float(np.mean(values)),
                     'seconds':time.perf_counter()-start,'reused_duplicate_candidate':reused})
        return float(np.mean(values))
    if method == 'grid':
        for trial,pars in enumerate(grid):
            evaluate(pars,trial)
        best = max(rows,key=lambda r:r['score'])
        params = best['params']
    else:
        def objective(trial):
            pars = {k:trial.suggest_categorical(k,v) for k,v in domains(name,cls).items()}
            return evaluate(pars,trial.number)
        study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=min(3,len(grid))))
        study.optimize(objective,n_trials=len(grid))
        params = study.best_params
    dump(path,{'model':name,'method':method,'rows':rows,'best_params':params,'unique_candidates':len(seen),
               'requested_evaluations':len(grid),'actual_inner_fits':len(seen)*len(cache['inner']),
               'duplicate_policy':'Identical deterministic candidate scores reused within one search; evaluations remain in trace.'})
    outer = cache['outer']
    fitted = make_model(name,params,cls,seed,outer['groups'],channels,steps,calibration_cv=outer['calibration_cv'],ablation=ablation)
    if name in NEURAL+RECURRENT:
        fitted.fit(outer['X'],outer['y'],validation=(outer['V'],(outer['actual']-outer['centre'])/outer['scale']))
    else:
        fitted.fit(outer['X'],outer['y'])
    return fitted,params


def prepare_jobs(root,out):
    root,out = Path(root),Path(out)
    datasets = {}
    for ds in ['house','adult','nba','students']:
        frame,meta = load_data(root,ds)
        datasets[ds] = (frame,meta['features'],meta['target'])
    frame,features = build_sequence(root,out/'nba_sequence')
    datasets['nba_sequence'] = (frame,features,'pts')
    jobs = []
    for ds,(frame,features,target) in datasets.items():
        cls = ds=='adult'
        variants = {'Full':[]}
        variants.update(SEQ_ABLATIONS if ds=='nba_sequence' else ABLAT[ds])
        outer_splits = seq_splits(frame) if ds=='nba_sequence' else splits(frame,ds,3)
        for fold,(a,b) in enumerate(outer_splits):
            train,valid = frame.iloc[a].reset_index(drop=True),frame.iloc[b].reset_index(drop=True)
            folder = out/ds/f'fold{fold}'
            folder.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({'row_id':frame._row_id,'partition':np.select([frame.index.isin(a),frame.index.isin(b)],['train','validation'],default='unused')}).to_csv(folder/'split.csv',index=False)
            cv = seq_splits(train) if ds=='nba_sequence' else splits(train,ds,3)
            dump(folder/'inner_splits.json',[{'train_row_ids':train.iloc[i]._row_id.tolist(),'validation_row_ids':train.iloc[j]._row_id.tolist()} for i,j in cv])
            for variant,removed in variants.items():
                cols = [c for c in features if (not any(c.endswith('_'+field) for field in removed) if ds=='nba_sequence' else c not in removed)]
                steps = 3
                if ds=='nba_sequence' and variant=='No oldest season':
                    cols = [c for c in cols if not c.startswith('lag3_')]
                    steps = 2
                channels = len(cols)//steps if ds=='nba_sequence' else 0
                cache_path = folder/variant.replace(' ','_')/'prepared.joblib'
                cache_path.parent.mkdir(parents=True,exist_ok=True)
                if not cache_path.exists():
                    cache = {'outer':prepare(train,valid,cols,target,cls),
                             'inner':[prepare(train.iloc[i],train.iloc[j],cols,target,cls) for i,j in cv]}
                    joblib.dump(cache,cache_path,compress=1)
                models = SEQUENCE_MODELS if ds=='nba_sequence' else MODELS
                # Mean is a fixed reference. Persistence has no fitted hyperparameter.
                for model in models:
                    methods = ['fixed'] if model in ['Mean','Persistence'] else ['grid','tpe'] if variant=='Full' else ['grid']
                    if variant!='Full' and model in ['Mean','Persistence']:
                        continue
                    for method in methods:
                        jobs.append({'dataset':ds,'fold':fold,'variant':variant,'model':model,'method':method,
                                     'cls':cls,'channels':channels,'steps':steps,'cache_path':str(cache_path),
                                     'folder':str(cache_path.parent),'seed':42+fold*100})
                if variant=='Full' and ds!='nba_sequence':
                    for model,ablation in ARCH_ABLATIONS.items():
                        jobs.append({'dataset':ds,'fold':fold,'variant':ablation,'model':model,'method':'grid','cls':cls,
                                     'channels':0,'steps':3,'cache_path':str(cache_path),'folder':str(folder/ablation.replace(' ','_')),
                                     'seed':42+fold*100,'architecture_ablation':ablation})
    return jobs


def run_job(task):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    warnings.filterwarnings('ignore',message='Found unknown categories')
    folder = Path(task['folder'])/(task['model']+'_'+task['method'])
    folder.mkdir(exist_ok=True,parents=True)
    if (folder/'complete.json').exists():
        return {**task,'cached':True}
    start = time.perf_counter()
    cache = joblib.load(task['cache_path'])
    o = cache['outer']
    name,cls = task['model'],task['cls']
    if task['method']=='fixed':
        model,params = None,{}
        if name=='Mean':
            train_p = np.full(len(o['y']),o['y'].mean())*o['scale']+o['centre']
            valid_p = np.full(len(o['actual']),train_p[0])
        else:
            index = o['feature_names'].index('lag1_pts')
            numeric = o['pre'].named_transformers_['num']
            scaler = numeric.named_steps['scale']
            train_p = o['X'][:,index]*scaler.scale_[index]+scaler.mean_[index]
            valid_p = o['V'][:,index]*scaler.scale_[index]+scaler.mean_[index]
    else:
        model,params = search_model(name,task['method'],cache,cls,task['seed'],task['channels'],task['steps'],folder/'search.json',task.get('architecture_ablation'))
        train_p = model_predict(model,o['X'],cls,name)*o['scale']+o['centre']
        valid_p = model_predict(model,o['V'],cls,name)*o['scale']+o['centre']
        joblib.dump(model,folder/'model.joblib',compress=1)
        if hasattr(model,'history'):
            pd.DataFrame(model.history).to_csv(folder/'training_curve.csv',index=False)
    assert np.isfinite(valid_p).all() and np.isfinite(train_p).all()
    rows = []
    for part,y,p in [('train',o['y']*o['scale']+o['centre'],train_p),('validation',o['actual'],valid_p)]:
        rows.append({'dataset':task['dataset'],'fold':task['fold'],'variant':task['variant'],'model':name,'method':task['method'],'partition':part,'n':len(y),**get_metric(y,p,cls)})
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame({'row_id':o['validation_ids'],'actual':o['actual'],'predicted':valid_p}).to_csv(folder/'predictions.csv',index=False)
    dump(folder/'complete.json',{'task':task,'selected_params':params,'seconds':time.perf_counter()-start,
                                'parameter_count':getattr(model,'parameter_count',None),
                                'model_sha256':sha(folder/'model.joblib') if model is not None else None,
                                'prediction_sha256':sha(folder/'predictions.csv'),'metrics_sha256':sha(folder/'metrics.csv')})
    return {**task,'seconds':time.perf_counter()-start,'cached':False}
