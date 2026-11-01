"""Nested evaluation on all four course datasets; no language-model calls."""
from pathlib import Path
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedGroupKFold, TimeSeriesSplit, GridSearchCV, ParameterGrid, cross_val_score
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, f1_score, brier_score_loss, log_loss
from tablelab.catalog import Catalog, SPECS
from tablelab.engine import predictor_fingerprints

SEED=42
OUTER=3
INNER=3
TRIALS=12
ABLAT={
 'house': {'No neighborhood':['Neighborhood'],'No size':['GrLivArea','TotalBsmtSF'],'No quality':['OverallQual'],'No year':['YearBuilt'],'Area + quality':['YearBuilt','TotalBsmtSF','GarageCars','Neighborhood']},
 'adult': {'No education':['education-num'],'No work':['hours-per-week','workclass','occupation'],'No marital':['marital-status'],'No age':['age']},
 'nba': {'No age':['age'],'No physique':['player_height','player_weight'],'No draft':['draft_year']},
 'students': {'No gender':['gender'],'No parental education':['parental level of education'],'No lunch':['lunch'],'No preparation':['test preparation course']}}

def dump(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)),encoding='utf-8')

def load_data(root,ds):
 df,entry=Catalog(root/'codes/data').load(ds)
 raw=len(df)
 if ds=='adult': df=df.loc[df['_source_split']=='train'].copy()
 target=SPECS[ds].targets[0]
 missing=int(df[target].isna().sum()); df=df.loc[df[target].notna()].reset_index(drop=True)
 return df,{'raw_rows':raw,'eligible_rows':len(df),'missing_target_excluded':missing,'source_sha256':entry['sha256'],'features':list(SPECS[ds].defaults),'target':target}

def splits(df,ds,n):
 if ds=='nba':
  seasons=np.array(sorted(df.season.unique()))
  result=[]
  for tr,va in TimeSeriesSplit(n_splits=n).split(seasons):
   a=np.flatnonzero(df.season.isin(seasons[tr]));b=np.flatnonzero(df.season.isin(seasons[va]));result.append((a,b))
  return result
 if ds=='adult':
  cols=[c for c in df if not c.startswith('_') and c not in ('income','income_gt_50k')]
  groups=predictor_fingerprints(df[cols])
  return list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=SEED).split(df,df.income_gt_50k,groups))
 return list(KFold(n_splits=n,shuffle=True,random_state=SEED).split(df))

def preprocess(X):
 nums=list(X.select_dtypes(include='number').columns); cats=[c for c in X if c not in nums]
 blocks=[]
 if nums: blocks.append(('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
 if cats: blocks.append(('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='__missing__')),('encode',OneHotEncoder(handle_unknown='ignore',drop='first',max_categories=30,sparse_output=False))]),cats))
 return ColumnTransformer(blocks,remainder='drop',verbose_feature_names_out=True)

def estimator(name,cls):
 common=dict(random_state=SEED)
 if name=='Mean':return DummyClassifier(strategy='prior') if cls else DummyRegressor()
 if name=='Linear':return LogisticRegression(C=1,max_iter=3000,**common) if cls else LinearRegression()
 if name=='Ridge':return LogisticRegression(C=.1,max_iter=3000,**common) if cls else Ridge(alpha=10)
 if name=='Sparse':return LogisticRegression(C=.1,solver='liblinear',penalty='l1',max_iter=3000,**common) if cls else ElasticNet(alpha=10,l1_ratio=.5,max_iter=10000,**common)
 if name=='Tree':return (DecisionTreeClassifier if cls else DecisionTreeRegressor)(**common)
 if name=='RF':return (RandomForestClassifier if cls else RandomForestRegressor)(n_estimators=64,max_depth=12,min_samples_leaf=5,n_jobs=1,**common)
 if name=='ExtraTrees':return (ExtraTreesClassifier if cls else ExtraTreesRegressor)(n_estimators=64,max_depth=12,min_samples_leaf=5,n_jobs=1,**common)
 if name=='HGB':return (HistGradientBoostingClassifier if cls else HistGradientBoostingRegressor)(max_iter=100,early_stopping=False,**common)
 if name=='KNN':return (KNeighborsClassifier if cls else KNeighborsRegressor)(n_neighbors=15,n_jobs=1)
 raise ValueError(name)

def pipeline(X,name,cls):return Pipeline([('pre',preprocess(X)),('model',estimator(name,cls))])

def space(name):
 if name=='RF':return {'model__max_depth':[4,8,None],'model__min_samples_leaf':[3,10],'model__max_features':[.7,1.]}
 return {'model__max_leaf_nodes':[7,15,31],'model__l2_regularization':[0.,1.],'model__learning_rate':[.05,.1]}

def metric(y,p,cls):
 if cls:
  return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),brier=brier_score_loss(y,p),log_loss=log_loss(y,p),f1=f1_score(y,p>=.5))
 return dict(rmse=float(np.sqrt(mean_squared_error(y,p))),mae=mean_absolute_error(y,p),r2=r2_score(y,p))

def prediction(pipe,X,cls):return pipe.predict_proba(X)[:,1] if cls else pipe.predict(X)

def search(pipe,X,y,cv,cls,method,fold,out):
 import optuna
 optuna.logging.set_verbosity(optuna.logging.WARNING)
 grid=space(pipe.named_steps['model'].__class__.__name__.startswith('RandomForest') and 'RF' or 'HGB')
 scoring='roc_auc' if cls else 'neg_root_mean_squared_error'
 rows=[]; start=time.perf_counter()
 if method=='grid':
  gs=GridSearchCV(pipe,grid,scoring=scoring,cv=cv,n_jobs=1,refit=True,error_score='raise',return_train_score=True)
  gs.fit(X,y)
  for j,p in enumerate(gs.cv_results_['params']):rows.append(dict(trial=j,score=float(gs.cv_results_['mean_test_score'][j]),seconds=float(gs.cv_results_['mean_fit_time'][j]*len(cv)),params=p))
  fitted=gs.best_estimator_;params=gs.best_params_
 else:
  def objective(trial):
   pars={k:trial.suggest_categorical(k,v) for k,v in grid.items()}
   t=time.perf_counter();value=float(cross_val_score(clone(pipe).set_params(**pars),X,y,scoring=scoring,cv=cv,n_jobs=1,error_score='raise').mean())
   rows.append(dict(trial=trial.number,score=value,seconds=time.perf_counter()-t,params=pars));return value
  study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED+fold,n_startup_trials=4))
  study.optimize(objective,n_trials=TRIALS)
  params=study.best_params;fitted=clone(pipe).set_params(**params).fit(X,y)
 dump(out,{'method':method,'rows':rows,'best_params':params,'total_seconds':time.perf_counter()-start,'unique_candidates':len({json.dumps(x['params'],sort_keys=True) for x in rows})})
 return fitted,params

def subset(df,ds,fraction):
 if fraction==1:return np.arange(len(df))
 if ds=='nba':
  seasons=sorted(df.season.unique());return np.flatnonzero(df.season.isin(seasons[:max(2,int(len(seasons)*fraction))]))
 if ds=='adult':
  cols=[c for c in df if not c.startswith('_') and c not in ('income','income_gt_50k')]
  group=predictor_fingerprints(df[cols]);unique=np.sort(group.unique());rng=np.random.default_rng(SEED);rng.shuffle(unique)
  return np.flatnonzero(group.isin(unique[:max(2,int(len(unique)*fraction))]))
 return np.random.default_rng(SEED).permutation(len(df))[:max(60,int(len(df)*fraction))]

def explain(fit,X,df,ds,out,fold):
 import shap
 pre=fit.named_steps['pre'];model=fit.named_steps['model'];Z=pre.transform(X)
 explainer=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw')
 ex=explainer(Z,check_additivity=True);values=np.asarray(ex.values)
 names=pre.get_feature_names_out();mapping=[]
 for col in X:
  mapping.append([j for j,n in enumerate(names) if n==f'num__{col}' or n.startswith(f'cat__{col}_')])
 grouped=np.column_stack([values[:,ids].sum(axis=1) for ids in mapping])
 base=np.asarray(ex.base_values).reshape(-1)
 raw=model._raw_predict(Z).reshape(-1) if ds=='adult' else model.predict(Z)
 error=float(np.max(np.abs(base+grouped.sum(axis=1)-raw)))
 assert error<1e-5,(ds,error)
 frame=pd.DataFrame(grouped,columns=X.columns);frame.insert(0,'row_id',df['_row_id'].values);frame.insert(1,'fold',fold);frame['base']=base;frame['raw_prediction']=raw
 frame.to_csv(out/'shap.csv',index=False)
 X.assign(row_id=df['_row_id'].values).to_csv(out/'shap_features.csv',index=False)
 dump(out/'shap_audit.json',{'max_additivity_error':error,'n_explained':len(X),'model':'HGB_grid','output':'log_odds' if ds=='adult' else 'target_units','dependence':'tree_path_dependent','background':'training tree path counts','encoding':'sum encoded columns per original field before taking absolute values'})

def run_dataset(root,ds):
 from threadpoolctl import threadpool_limits
 threadpool_limits(1)
 root=Path(root); out=root/'artifacts/research'/ds;out.mkdir(parents=True,exist_ok=True)
 df,meta=load_data(root,ds); features=meta['features'];target=meta['target'];cls=ds=='adult'
 dump(out/'data_audit.json',meta)
 print(f'{ds}: {len(df)} rows; starting',flush=True)
 for fold,(tr,va) in enumerate(splits(df,ds,OUTER)):
  folder=out/f'fold{fold}';folder.mkdir(exist_ok=True)
  if (folder/'complete.json').exists():continue
  train=df.iloc[tr].reset_index(drop=True);valid=df.iloc[va].reset_index(drop=True)
  cv=splits(train,ds,INNER);X=train[features];Y=train[target];V=valid[features];yv=valid[target]
  st=pd.DataFrame({'row_id':df._row_id,'partition':'unused'})
  st.loc[tr,'partition']='train';st.loc[va,'partition']='validation';st.to_csv(folder/'split.csv',index=False)
  dump(folder/'inner_splits.json',[{'train_row_ids':train.iloc[a]._row_id.tolist(),'validation_row_ids':train.iloc[b]._row_id.tolist()} for a,b in cv])
  results=[];preds=[];fits={}
  for name in ['Mean','Linear','Ridge','Sparse','Tree','RF','ExtraTrees','HGB','KNN','RF_grid','RF_tpe','HGB_grid','HGB_tpe']:
   t=time.perf_counter()
   if '_' in name:
    family,method=name.split('_');fit,params=search(pipeline(X,family,cls),X,Y,cv,cls,method,fold,folder/f'{name}_search.json')
   else:
    fit=pipeline(X,name,cls).fit(X,Y);params={}
   fits[name]=fit
   for part,xx,yy in [('train',X,Y),('validation',V,yv)]:
    pp=prediction(fit,xx,cls);results.append(dict(dataset=ds,fold=fold,model=name,partition=part,n=len(yy),seconds=time.perf_counter()-t,**metric(yy,pp,cls)))
    if part=='validation':preds.extend(dict(row_id=r,model=name,actual=float(y),predicted=float(p),fold=fold) for r,y,p in zip(valid._row_id,yy,pp))
   joblib.dump(fit,folder/f'{name}.joblib',compress=3)
   pd.DataFrame(results).to_csv(folder/'metrics.csv',index=False)
   print(f'{ds} fold {fold} {name}: {results[-1]}',flush=True)
  pd.DataFrame(preds).to_csv(folder/'predictions.csv',index=False)
  diagnostics=[]
  for fraction in [.25,.5,.75,1.]:
   ix=subset(train,ds,fraction)
   fit=clone(fits['HGB_grid']).fit(X.iloc[ix],Y.iloc[ix])
   for part,xx,yy in [('train',X.iloc[ix],Y.iloc[ix]),('validation',V,yv)]:diagnostics.append(dict(dataset=ds,fold=fold,kind='learning',fraction=fraction,n_train=len(ix),partition=part,**metric(yy,prediction(fit,xx,cls),cls)))
  for depth in [1,2,4,8,None]:
   fit=pipeline(X,'Tree',cls).set_params(model__max_depth=depth).fit(X,Y)
   for part,xx,yy in [('train',X,Y),('validation',V,yv)]:diagnostics.append(dict(dataset=ds,fold=fold,kind='complexity',depth=depth or -1,partition=part,**metric(yy,prediction(fit,xx,cls),cls)))
  pd.DataFrame(diagnostics).to_csv(folder/'diagnostics.csv',index=False)
  ablations=[]
  for label,removed in ABLAT[ds].items():
   cols=[c for c in features if c not in removed]
   fit,params=search(pipeline(X[cols],'HGB',cls),X[cols],Y,cv,cls,'grid',fold,folder/f'ablation_{label.replace(" ","_")}.json')
   pp=prediction(fit,V[cols],cls)
   ablations.append(dict(dataset=ds,fold=fold,variant=label,n_validation=len(yv),**metric(yv,pp,cls)))
  pd.DataFrame(ablations).to_csv(folder/'ablations.csv',index=False)
  explain(fits['HGB_grid'],V,valid,ds,folder,fold)
  dump(folder/'complete.json',{'models':len(fits),'ablations':len(ablations),'n_train':len(train),'n_validation':len(valid)})
  print(f'{ds} fold {fold} complete',flush=True)
 return ds
