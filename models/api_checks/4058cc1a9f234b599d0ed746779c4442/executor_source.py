"""固定分析执行器。所有输入均先校验；不使用 eval、exec、shell 或动态 SQL。"""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import time
import uuid
import warnings
import importlib.metadata
import platform
import sys
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
    balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix)
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
from .catalog import Catalog, SPECS, DataError
from .contracts import Plan

SEED=42

def predictor_fingerprints(frame):
    """仅对非目标原始字段生成稳定指纹；不将其解释为个人身份。"""
    rows=json.loads(frame.to_json(orient='values',force_ascii=False,double_precision=15))
    return pd.Series([hashlib.sha256(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest() for row in rows],index=frame.index)

def runtime_fingerprint():
    names=['numpy','pandas','scikit-learn','scipy','matplotlib','pydantic','fonttools']
    files=sorted(Path(__file__).parent.glob('*.py'))
    digest=hashlib.sha256()
    for file in files:
        digest.update(file.name.encode());digest.update(file.read_bytes())
    return {'package_sha256':digest.hexdigest(),'dependencies':{n:importlib.metadata.version(n) for n in names},'python':sys.version,'platform':platform.platform()}

def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def table_records(df):
    return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=12))

def require_columns(df, columns):
    invalid=[c for c in columns if c not in df or c.startswith('_')]
    if invalid: raise DataError(f'字段不存在或不能直接操作：{invalid}')

def require_numeric(df, columns):
    require_columns(df,columns)
    for c in columns:
        if not is_numeric_dtype(df[c]): raise DataError(f'{c} 不是数值字段。')
        if np.isinf(df[c].dropna().to_numpy(dtype=float)).any(): raise DataError(f'{c} 含无穷值。')

def apply_filters(df, filters):
    mask=pd.Series(True,index=df.index)
    for rule in filters:
        require_columns(df,[rule.column]); s=df[rule.column]
        if rule.op=='is_missing': m=s.isna()
        elif rule.op=='not_missing': m=s.notna()
        else:
            if is_numeric_dtype(s):
                vals=rule.value if isinstance(rule.value,list) else [rule.value]
                if any(not isinstance(v,(int,float)) or isinstance(v,bool) for v in vals):
                    raise DataError(f'{rule.column} 的筛选值必须是数值，不能用字符串或布尔值。')
            elif rule.op in ('gt','ge','lt','le'):
                raise DataError('分类字段仅支持 eq、ne、in 和缺失值筛选。')
            if rule.op=='in': m=s.isin(rule.value)
            elif rule.op=='eq': m=s.eq(rule.value)
            elif rule.op=='ne': m=s.ne(rule.value)
            elif rule.op=='gt': m=s.gt(rule.value)
            elif rule.op=='ge': m=s.ge(rule.value)
            elif rule.op=='lt': m=s.lt(rule.value)
            else: m=s.le(rule.value)
            m=m & s.notna()
        mask=mask & m
    out=df.loc[mask].copy()
    if out.empty: raise DataError('筛选后没有记录；请核对条件和类别取值。')
    return out

def aggregate_frame(df, plan):
    require_columns(df,plan.group_by+plan.columns)
    if plan.metric!='count': require_numeric(df,plan.columns)
    measure=plan.columns[0] if plan.columns else None
    if plan.group_by:
        grouped=df.groupby(plan.group_by,dropna=False,observed=True,sort=True)
        result=grouped.size().rename('n_rows').to_frame()
        if measure:
            result['n_valid']=grouped[measure].count()
            result['value']=(grouped[measure].count() if plan.metric=='count' else grouped[measure].sum(min_count=1) if plan.metric=='sum' else grouped[measure].agg(plan.metric))
        else:
            result['n_valid']=result['n_rows']; result['value']=result['n_rows']
        result=result.reset_index()
    else:
        s=df[measure] if measure else None
        value=(s.count() if measure else len(df)) if plan.metric=='count' else (s.sum(min_count=1) if plan.metric=='sum' else s.agg(plan.metric))
        result=pd.DataFrame([{'group':'全部样本','n_rows':len(df),'n_valid':int(s.count()) if measure else len(df),'value':value}])
    return result.sort_values('value',ascending=not plan.descending,kind='stable',na_position='last').reset_index(drop=True)

@dataclass
class Fitted:
    dataset: str
    plan: Plan
    pipeline: object
    numeric: list[str]
    categories: dict[str,list]
    ranges: dict[str,tuple[float,float]]
    provenance: dict

class Engine:
    def __init__(self,catalog: Catalog,out: Path):
        self.catalog=catalog
        self.out=Path(out).resolve(); self.out.mkdir(parents=True,exist_ok=True)
        self.models={}

    def execute(self,plan: Plan,question='',evidence=None):
        if plan.operation in ('unsupported','clarify'):
            return {'status':plan.operation,'summary':plan.message,'table':[],'notes':[],'files':[],'plan':plan.model_dump()}
        # 与操作无关的非默认字段不能被悄悄忽略。
        if plan.features and plan.operation!='train': raise DataError('只有 train 接受 features。')
        if plan.target and plan.operation!='train': raise DataError('只有 train 接受 target。')
        if plan.record and plan.operation!='predict': raise DataError('只有 predict 接受 record。')
        if plan.group_by and not (plan.operation=='aggregate' or (plan.operation=='plot' and plan.chart=='bar')):
            raise DataError('此操作不接受 group_by；分组统计请使用 aggregate。')
        if plan.operation in ('predict','explain') and plan.filters: raise DataError('字段解释和单条预测不接受样本筛选条件。')
        df,entry=self.catalog.load(plan.dataset)
        run_id=uuid.uuid4().hex; folder=self.out/run_id; folder.mkdir()
        start=time.perf_counter()
        write_json(folder/'plan.json',plan.model_dump())
        write_json(folder/'evidence.json',evidence or [])
        source=Path(__file__).read_text(encoding='utf-8')
        (folder/'executor_source.py').write_text(source,encoding='utf-8')
        provenance={**entry,'dataset':plan.dataset,'engine_sha256':hashlib.sha256(source.encode()).hexdigest(),
                    'run_id':run_id,'question':question,'rows_before_filter':len(df),'random_seed':SEED,
                    'runtime':runtime_fingerprint()}
        notes=[SPECS[plan.dataset].warning]
        if entry['source_kind']=='synthetic_test': notes.insert(0,'当前是合成测试数据，不是作业原始数据；所有数值仅用于软件验证。')
        if entry['source_kind']!='synthetic_test': notes.append(entry.get('source_note','课程数据副本；哈希一致不等于原网站认证。'))
        selected=apply_filters(df,plan.filters)
        provenance['rows_after_filter']=len(selected)
        result={'status':'ok','run_id':run_id,'source_kind':entry['source_kind'],'plan':plan.model_dump(),
                'notes':notes,'files':[],'table':[],'metrics':{}}
        try:
            if plan.operation=='explain':
                require_columns(df,plan.columns)
                columns=plan.columns or list(SPECS[plan.dataset].descriptions)
                table=pd.DataFrame([{'column':c,'definition':SPECS[plan.dataset].descriptions.get(c,'未提供已核实的字段释义；请查原始数据字典。')} for c in columns if c in df])
                result['summary']='以下释义来自本地字段说明；未检索到的定义不补写。'
            elif plan.operation=='describe':
                columns=plan.columns or [c for c in selected if not c.startswith('_')]
                require_columns(selected,columns); rows=[]
                for c in columns:
                    s=selected[c]; r={'column':c,'dtype':str(s.dtype),'n_valid':int(s.count()),'missing':int(s.isna().sum()),'unique':int(s.nunique())}
                    if is_numeric_dtype(s):
                        require_numeric(selected,[c]); r.update(mean=s.mean(),std=s.std(),min=s.min(),q25=s.quantile(.25),median=s.median(),q75=s.quantile(.75),max=s.max())
                    rows.append(r)
                table=pd.DataFrame(rows)
                result['summary']=f'在筛选后的 {len(selected):,} 条记录上计算描述统计。缺失值计数单独报告；数值统计使用该字段的非缺失记录。'
            elif plan.operation=='aggregate':
                table=aggregate_frame(selected,plan)
                result['summary']=f'聚合使用筛选后的全部 {len(selected):,} 条记录，共 {len(table):,} 个结果组。n_rows 是组内行数，n_valid 是实际参与该统计量的非缺失数。'
            elif plan.operation=='correlation':
                require_numeric(selected,plan.columns); pair=selected[plan.columns].dropna()
                if len(pair)<3 or any(pair[c].nunique()<2 for c in pair): raise DataError('有效样本不足或字段为常数，相关系数未定义。')
                r=float(pair.iloc[:,0].corr(pair.iloc[:,1]))
                table=pd.DataFrame([{'x':plan.columns[0],'y':plan.columns[1],'pearson_r':r,'n_complete':len(pair),'excluded_missing':len(selected)-len(pair)}])
                result['summary']=f'两字段均非缺失的 {len(pair):,} 条记录，皮尔逊相关系数为 {r:.4f}。这不是因果效应，也没有进行显著性检验。'
            elif plan.operation=='plot':
                from .plotting import data_plot
                require_numeric(selected,plan.columns)
                if plan.chart=='bar':
                    table=aggregate_frame(selected,plan)
                else:
                    table=selected[['_row_id']+plan.columns].dropna()
                    if table.empty: raise DataError('绘图没有完整的数值记录。')
                data_plot(table,plan,folder/'figure.pdf',entry['source_kind'])
                result['files'].append('figure.pdf')
                result['summary']=f'已导出 PDF。原筛选样本 {len(selected):,} 条；图中使用 {len(table):,} 个完整点或汇总组。柱形图仅显示排序后的前 {plan.limit} 组，完整汇总保留在 CSV。'
            elif plan.operation=='train':
                table,fitinfo=self._train(selected,plan,folder,run_id,provenance)
                result.update(fitinfo)
                result['notes']=notes+fitinfo['notes']
            else:
                table,info=self._predict(plan,entry)
                result.update(info)
                result['notes']=notes+info['notes']
            table.to_csv(folder/'table.csv',index=False,encoding='utf-8-sig')
            result['table']=table_records(table.head(plan.limit if plan.operation=='aggregate' else 100))
            result['table_rows_total']=len(table)
            result['files'] += ['table.csv','plan.json','evidence.json','executor_source.py','provenance.json','reproduce.py','result.json']
            provenance['elapsed_seconds']=round(time.perf_counter()-start,6)
            result['provenance']=provenance
            write_json(folder/'provenance.json',provenance)
            (folder/'reproduce.py').write_text(
                '"""在项目根目录运行此文件；使用相同 data/manifest.json 与已核对的执行器。"""\n'
                'from pathlib import Path\nimport json,hashlib,os,sys\nsys.path.insert(0,str(Path.cwd()))\n'
                'from tablelab import engine as module\nfrom tablelab.engine import Engine,runtime_fingerprint\n'
                'from tablelab.catalog import Catalog\nfrom tablelab.contracts import Plan\n'
                'here=Path(__file__).resolve().parent\nmeta=json.loads((here/"provenance.json").read_text())\n'
                'assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==meta["engine_sha256"], "执行器版本已改变"\n'
                'current=runtime_fingerprint()\nassert current["package_sha256"]==meta["runtime"]["package_sha256"], "分析模块版本已改变"\nassert current["dependencies"]==meta["runtime"]["dependencies"], "依赖版本已改变"\n'
                'root=Path(os.environ.get("TABLELAB_DATA_ROOT", "data"))\n'
                'assert hashlib.sha256((root/"processed"/(meta["dataset"]+".csv")).read_bytes()).hexdigest()==meta["sha256"], "数据已改变"\n'
                'plan=Plan.model_validate_json((here/"plan.json").read_text())\n'
                'if plan.operation=="predict": raise RuntimeError("单条预测依赖会话中的已训练模型；先重放对应训练方案再预测。")\n'
                'print(json.dumps(Engine(Catalog(root),Path("artifacts/replay")).execute(plan),ensure_ascii=False,indent=2))\n',encoding='utf-8')
            write_json(folder/'result.json',result)
            return result
        except (DataError,ValueError) as exc:
            write_json(folder/'error.json',{'status':'failed','error':str(exc),'provenance':provenance})
            raise

    def _train(self,df,plan,folder,run_id,provenance):
        spec=SPECS[plan.dataset]
        if plan.target not in spec.targets: raise DataError(f'此数据集支持的目标是 {list(spec.targets)}。')
        require_columns(df,plan.features+[plan.target]); require_numeric(df,[plan.target])
        blocked=set(spec.excluded)|set(c for c in df if c.startswith('_'))
        if any(c in blocked or c.startswith('Unnamed:') for c in plan.features): raise DataError('特征含目标、标识符或当前不支持的样本权重。')
        if plan.dataset=='adult' and {'education','education-num'}<=set(plan.features):
            raise DataError('education 与 education-num 重复编码教育程度，请选择其中一个。')
        classification=plan.dataset=='adult'
        if classification and plan.algorithm=='linear': raise DataError('收入二元分类请使用 logistic 或 random_forest。')
        if not classification and plan.algorithm=='logistic': raise DataError('当前连续目标不支持 logistic；不能擅自离散化目标。')
        notes=[]
        if plan.context=='prospective':
            contemporaneous={'students':{'math score','reading score','writing score'},'nba':{'pts','reb','ast','gp','net_rating','usg_pct','ts_pct','ast_pct','oreb_pct','dreb_pct'}}
            if set(plan.features)&contemporaneous.get(plan.dataset,set()):
                raise DataError('预测时尚不可用的同期结果不能进入 prospective 模型；改用提前可观测字段，或明确选择 retrospective 同期估计。')
        else:
            notes.append('本次 context=retrospective；模型用于所列特征已知时的同期估计，不宣称前瞻预测。')
        if plan.dataset=='adult':
            if '_source_split' not in df: raise DataError('Adult 缺少原始训练/测试来源标识。')
            excluded=int((df['_source_split']!='train').sum()); df=df.loc[df['_source_split']=='train'].copy()
            notes.append(f'已排除 {excluded:,} 条原始测试记录；仅在 adult.data 内做交互训练/验证。')
        before=len(df); df=df.dropna(subset=[plan.target]).copy()
        provenance['rows_missing_target_excluded']=before-len(df)
        if len(df)<100: raise DataError('有效目标少于 100 条；当前演示拒绝在此样本规模上训练。')
        if len(df)>100000: raise DataError('训练样本超过 100000 条资源上限；请显式筛选，程序不会静默抽样。')
        X=df[plan.features]; y=df[plan.target]
        if classification and set(y.unique())!={0,1}: raise DataError('分类目标必须同时包含 0 和 1。')
        if not classification and y.nunique()<2: raise DataError('目标为常数，拒绝拟合。')
        if plan.dataset=='nba' and plan.split=='random': raise DataError('NBA 不支持逐行随机划分；请选择 time 或 group。')
        positions=np.arange(len(df))
        if plan.split=='random':
            if classification and y.value_counts().min()<10: raise DataError('分类样本中最少类别不足 10 条。')
            if classification:
                group_columns=[c for c in df if not c.startswith('_') and c not in ('income','income_gt_50k')]
                groups=predictor_fingerprints(df[group_columns])
                train,val=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
                assert set(groups.iloc[train]).isdisjoint(groups.iloc[val]), '重复预测字段跨越训练与验证。'
                split_info={'kind':'stratified_predictor_groups','seed':SEED,'fold':0,'n_folds':5,'fingerprint_columns':group_columns,'group_overlap':0}
                notes.append('Adult 按全部原始非目标字段的指纹分组后分层划分；相同预测字段记录保留，但不跨越训练与验证。指纹不是个人身份。')
            else:
                train,val=train_test_split(positions,test_size=.2,random_state=SEED)
                split_info={'kind':'random','seed':SEED}
        elif plan.split=='time':
            field='season' if plan.dataset=='nba' else 'YrSold' if plan.dataset=='house' else None
            if field is None or field not in df: raise DataError('当前数据集缺少已注册的时间字段。')
            if field=='season':
                if not df[field].astype(str).str.fullmatch(r'\d{4}-\d{2}').all(): raise DataError('赛季格式不符合 YYYY-YY。')
                time_values=df[field].str.slice(0,4).astype(int)
            else: time_values=df[field]
            if time_values.isna().any(): raise DataError('时间字段含缺失值。')
            values=sorted(time_values.unique())
            if len(values)<3: raise DataError('时间划分至少需要 3 个不同年份/赛季。')
            boundary=values[-max(1,math.ceil(len(values)*.2))]
            train=np.flatnonzero(time_values.to_numpy()<boundary); val=np.flatnonzero(time_values.to_numpy()>=boundary)
            split_info={'kind':'time','field':field,'validation_start':int(boundary),'train_periods':[str(x) for x in sorted(df.iloc[train][field].unique())],'validation_periods':[str(x) for x in sorted(df.iloc[val][field].unique())]}
            if plan.dataset=='nba': notes.append('按后续赛季留出验证，不自动等于严格季前预测；本数据没有给出每个字段的采集时点。')
        else:
            if plan.dataset!='nba' or 'player_name' not in df: raise DataError('group 仅支持 NBA 的球员分组。')
            if df['player_name'].isna().any() or df['player_name'].nunique()<10: raise DataError('球员分组键缺失或少于 10 组。')
            train,val=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(X,y,groups=df['player_name']))
            if set(df.iloc[train]['player_name'])&set(df.iloc[val]['player_name']): raise AssertionError('球员跨组泄漏。')
            split_info={'kind':'group','field':'player_name','seed':SEED,'group_overlap':0}
            notes.append('按姓名分组，同名不同人可能被保守放在同组；缺少稳定球员 ID，未声称完成真实身份消歧。')
        if len(train)<60 or len(val)<20: raise DataError('划分后的训练/验证样本不足 60/20 条。')
        if set(train)&set(val): raise AssertionError('训练与验证行重叠。')
        Xtrain=X.iloc[train]; Xval=X.iloc[val]; ytrain=y.iloc[train]; yval=y.iloc[val]
        if classification and (set(ytrain)!={0,1} or set(yval)!={0,1} or yval.value_counts().min()<3):
            raise DataError('训练/验证必须各有两个类别，验证集中每类至少 3 条。')
        numeric=[c for c in plan.features if is_numeric_dtype(X[c])]
        categorical=[c for c in plan.features if c not in numeric]
        require_numeric(X,numeric)
        if any(Xtrain[c].notna().sum()==0 for c in plan.features): raise DataError('至少一个特征在训练子集中全缺失。')
        transformations=[]
        if numeric:
            transformations.append(('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric))
        if categorical:
            transformations.append(('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='__missing__')),
                ('encode',OneHotEncoder(handle_unknown='ignore',drop='first',max_categories=30,sparse_output=False))]),categorical))
        pre=ColumnTransformer(transformations,remainder='drop',verbose_feature_names_out=True)
        if plan.algorithm=='linear': model=LinearRegression()
        elif plan.algorithm=='logistic': model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1500,random_state=SEED)
        else:
            model=(RandomForestClassifier if classification else RandomForestRegressor)(n_estimators=120,max_depth=12,min_samples_leaf=5,n_jobs=1,random_state=SEED)
        pipe=Pipeline([('preprocess',pre),('model',model)])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            pipe.fit(Xtrain,ytrain)
        transform_audit={'numeric':numeric,'categorical':categorical,'model_parameters':model.get_params(),'n_train':len(train),'n_validation':len(val),'target_training_mean':float(ytrain.mean())}
        if numeric:
            transformer=pre.named_transformers_['num']
            transform_audit['imputation_medians']=dict(zip(numeric,transformer.named_steps['impute'].statistics_.tolist()))
            transform_audit['scaler_means']=dict(zip(numeric,transformer.named_steps['scale'].mean_.tolist()))
            transform_audit['scaler_variances']=dict(zip(numeric,transformer.named_steps['scale'].var_.tolist()))
        if categorical:
            enc=pre.named_transformers_['cat'].named_steps['encode']
            transform_audit['training_categories']={c:list(map(str,values)) for c,values in zip(categorical,enc.categories_)}
        transform_audit['transformed_feature_names']=pre.get_feature_names_out().tolist()
        if plan.algorithm=='linear': transform_audit['linear_rank']=int(model.rank_)
        write_json(folder/'transformation_audit.json',transform_audit)
        categories={c:sorted(Xtrain[c].dropna().astype(str).unique().tolist()) for c in categorical}
        for c in categorical:
            unknown=set(Xval[c].dropna().astype(str))-set(categories[c])
            if unknown: notes.append(f'{c} 验证集出现 {len(unknown)} 个训练时未见类别；编码为全零，不代表已学会这些类别。')
        with warnings.catch_warnings(record=True) as captured:
            pred=pipe.predict(Xval)
        for w in captured: notes.append(str(w.message))
        if classification:
            proba=pipe.predict_proba(Xval)[:,list(pipe.classes_).index(1)]
            pred=(proba>=.5).astype(int)
            baseline=DummyClassifier(strategy='prior').fit(Xtrain,ytrain)
            baseprob=baseline.predict_proba(Xval)[:,list(baseline.classes_).index(1)]
            metrics={'roc_auc':float(roc_auc_score(yval,proba)),'average_precision':float(average_precision_score(yval,proba)),
                'accuracy':float(accuracy_score(yval,pred)),'balanced_accuracy':float(balanced_accuracy_score(yval,pred)),
                'f1':float(f1_score(yval,pred,zero_division=0)),'brier':float(brier_score_loss(yval,proba)),
                'log_loss':float(log_loss(yval,proba,labels=[0,1])),
                'baseline_brier':float(brier_score_loss(yval,baseprob)), 'validation_positive_rate':float(yval.mean())}
            summary=f'训练 {len(train):,} 条，交互验证 {len(val):,} 条。ROC AUC={metrics["roc_auc"]:.4f}；F1={metrics["f1"]:.4f}；Brier={metrics["brier"]:.4f}。AUC 衡量排序，Brier 衡量概率误差，两者不能互相替代。'
            notes.append('分类阈值固定 0.5；概率未额外校准。本实现的逻辑回归含 L2 正则，不提供 p 值或因果解释。')
            predictions=pd.DataFrame({'row_id':df.iloc[val]['_row_id'].to_numpy(),'actual':yval.to_numpy(),'predicted_class':pred,'probability_1':proba})
            pd.DataFrame(confusion_matrix(yval,pred,labels=[0,1]),index=['actual_0','actual_1'],columns=['pred_0','pred_1']).to_csv(folder/'confusion.csv',encoding='utf-8-sig')
        else:
            baseline=DummyRegressor(strategy='mean').fit(Xtrain,ytrain).predict(Xval)
            metrics={'mae':float(mean_absolute_error(yval,pred)),'rmse':float(math.sqrt(mean_squared_error(yval,pred))),
                'r2':float(r2_score(yval,pred)),'baseline_rmse':float(math.sqrt(mean_squared_error(yval,baseline)))}
            summary=f'训练 {len(train):,} 条，交互验证 {len(val):,} 条。MAE={metrics["mae"]:.4f}；RMSE={metrics["rmse"]:.4f}；R²={metrics["r2"]:.4f}。训练均值基线 RMSE={metrics["baseline_rmse"]:.4f}。误差单位与 {plan.target} 一致。'
            predictions=pd.DataFrame({'row_id':df.iloc[val]['_row_id'].to_numpy(),'actual':yval.to_numpy(),'predicted':pred,'residual':yval.to_numpy()-pred})
        predictions.to_csv(folder/'validation_predictions.csv',index=False,encoding='utf-8-sig')
        split_table=pd.DataFrame({'row_id':df['_row_id'].to_numpy(),'partition':'train'})
        split_table.loc[val,'partition']='interactive_validation'
        if plan.dataset=='adult': split_table['predictor_fingerprint']=predictor_fingerprints(df[[c for c in df if not c.startswith('_') and c not in ('income','income_gt_50k')]]).to_numpy()
        split_table.to_csv(folder/'split.csv',index=False,encoding='utf-8-sig')
        provenance['split']={**split_info,'n_train':len(train),'n_validation':len(val),'n_eligible':len(df)}
        table=pd.DataFrame([{'metric':k,'value':v} for k,v in metrics.items()])
        table.to_csv(folder/'metrics.csv',index=False,encoding='utf-8-sig')
        from .plotting import diagnostic_plot
        diagnostic_plot(predictions,classification,folder/'diagnostic.pdf',provenance['source_kind'])
        files=['metrics.csv','validation_predictions.csv','split.csv','diagnostic.pdf','transformation_audit.json']
        if classification: files.append('confusion.csv')
        if plan.algorithm=='random_forest':
            imp=permutation_importance(pipe,Xval,yval,n_repeats=3,random_state=SEED,n_jobs=1,
                scoring='roc_auc' if classification else 'neg_root_mean_squared_error')
            explanation=pd.DataFrame({'feature':plan.features,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False)
            notes.append('置换重要性在完整交互验证集上计算，重复 3 次；标准差只是置换间波动，不是置信区间。')
            top=explanation.iloc[0]
            unit='AUC 下降值' if classification else 'RMSE 增加值'
            summary+=f' 本次置换重要性均值最大的特征为 {top["feature"]}，{unit}为 {top["importance_mean"]:.4f}；这不表示因果作用。'
        else:
            coefs=pipe.named_steps['model'].coef_.ravel()
            names=pipe.named_steps['preprocess'].get_feature_names_out()
            explanation=pd.DataFrame({'feature':names,'coefficient':coefs})
            notes.append('数值变量系数以训练子集一个标准差为单位；分类变量相对独热编码的参考类别。相关特征可能使系数不稳定。')
            if plan.algorithm=='linear' and model.rank_<len(names): notes.append('设计矩阵不满列秩；系数不唯一，不应逐项作结构性解释。')
            if categorical:
                encoder=pre.named_transformers_['cat'].named_steps['encode']
                refs={c:str(cats[i]) if i is not None else '(none)' for c,cats,i in zip(categorical,encoder.categories_,encoder.drop_idx_)}
                notes.append('分类参考类别：'+json.dumps(refs,ensure_ascii=False))
            provenance['intercept']=float(np.ravel(model.intercept_)[0])
            if numeric:
                first=numeric[0];coefficient=float(explanation.loc[explanation['feature']=='num__'+first,'coefficient'].iloc[0])
                effect='预测对数几率' if classification else f'预测 {plan.target}'
                summary+=f' 其余编码特征固定时，{first} 每增加训练样本的一个标准差，{effect}变化 {coefficient:.4f}；这是模型内的条件关联。'
        explanation.to_csv(folder/'explanation.csv',index=False,encoding='utf-8-sig');files.append('explanation.csv')
        notes.append('这些指标来自可反复查看的交互验证集，不是最终盲测结果；不要用比较次数挑选最好的数字后宣称泛化性能。')
        self.models[run_id]=Fitted(plan.dataset,plan,pipe,numeric,categories,{c:(float(Xtrain[c].min()),float(Xtrain[c].max())) for c in numeric},provenance.copy())
        while len(self.models)>4: del self.models[next(iter(self.models))]
        return table,{'metrics':metrics,'summary':summary,'notes':notes,'files':files,'model_id':run_id,
                      'explanation':table_records(explanation.head(30))}

    def _predict(self,plan,entry):
        if plan.model_id not in self.models: raise DataError('此模型不在当前会话中；重启后或模型被替换后需要重新训练。')
        fit=self.models[plan.model_id]
        if fit.dataset!=plan.dataset: raise DataError('不能跨数据集使用模型。')
        if entry['sha256']!=fit.provenance['sha256']: raise DataError('训练后数据已改变；必须重新训练。')
        if set(plan.record)!=set(fit.plan.features): raise DataError('预测输入必须恰好包含训练时的全部特征，缺失值用 null 明确表示。')
        notes=[]
        for c,v in plan.record.items():
            if v is None: notes.append(f'{c} 缺失，使用训练时保存的填补规则。');continue
            if c in fit.numeric:
                if not isinstance(v,(int,float)) or isinstance(v,bool): raise DataError(f'{c} 必须为数值。')
                lo,hi=fit.ranges[c]
                if not lo<=v<=hi: notes.append(f'{c} 超出训练观察范围 [{lo:g}, {hi:g}]；这是外推。')
            elif str(v) not in fit.categories[c]: notes.append(f'{c} 是未见类别；编码为全零，预测适用性有限。')
        X=pd.DataFrame([{c:(np.nan if plan.record[c] is None else plan.record[c]) for c in fit.plan.features}])
        if plan.dataset=='adult':
            value=float(fit.pipeline.predict_proba(X)[0,list(fit.pipeline.classes_).index(1)])
            table=pd.DataFrame([{'probability_gt_50k':value,'predicted_class_at_0_5':int(value>=.5)}])
            summary=f'本条输入的 >50K 类别预测概率为 {value:.4%}。这是样本训练模型的估计概率，不是年收入金额，也不保证概率已校准。'
        else:
            value=float(fit.pipeline.predict(X)[0]);table=pd.DataFrame([{'target':fit.plan.target,'prediction':value}])
            summary=f'{fit.plan.target} 的点预测为 {value:.4f}。未计算预测区间；数值适用范围取决于训练样本和输入特征。'
            if value<0: notes.append('模型给出负预测值；当前目标通常非负，该结果需审查，程序未自动裁剪。')
        notes.append('模型最初训练条件：'+fit.plan.model_dump_json())
        return table,{'summary':summary,'notes':notes,'model_id':plan.model_id,'training_provenance':fit.provenance}
