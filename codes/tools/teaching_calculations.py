"""重算讲解中的小例子，并用旧版保存的系数重建预测；不重训基准模型。"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tablelab.catalog import Catalog
from tablelab.contracts import Plan
from tablelab.engine import Engine
from tablelab.retrieval import Retriever

OUT = ROOT / 'verification' / 'editorial'
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / 'tables' / 'teaching'
TABLES.mkdir(parents=True, exist_ok=True)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def reconstruct(case, frame, classification=False):
    baseline = pd.read_csv(ROOT/'tables/real_data_baselines.csv')
    run = baseline.loc[baseline['case'].eq(case), 'run_id'].unique()
    assert len(run) == 1
    folder = ROOT/'artifacts/benchmarks'/run[0]
    audit = json.loads((folder/'transformation_audit.json').read_text())
    provenance = json.loads((folder/'provenance.json').read_text())
    pred = pd.read_csv(folder/'validation_predictions.csv')
    coef = pd.read_csv(folder/'explanation.csv').set_index('feature')['coefficient']
    data = frame.set_index('_row_id').loc[pred.row_id]
    terms = {}
    for name in audit['numeric']:
        sigma = np.sqrt(audit['scaler_variances'][name])
        values = data[name].fillna(audit['imputation_medians'][name]).to_numpy(float)
        terms['num__'+name] = (values-audit['scaler_means'][name])/sigma
    for name in audit['categorical']:
        values = data[name].fillna('__missing__').astype(str).to_numpy()
        for level in audit['training_categories'][name][1:]:
            key = 'cat__'+name+'_'+level
            assert key in coef.index, (case, key)
            terms[key] = (values == level).astype(float)
    design = pd.DataFrame(terms)[list(coef.index)]
    eta = provenance['intercept'] + design.to_numpy() @ coef.to_numpy()
    rebuilt = expit(eta) if classification else eta
    stored = pred['probability_1' if classification else 'predicted'].to_numpy()
    np.testing.assert_allclose(rebuilt, stored, rtol=1e-10, atol=1e-8)
    sample = pred.iloc[[0]].copy()
    sample['rebuilt'] = rebuilt[0]
    sample['linear_predictor'] = eta[0]
    sample.to_csv(TABLES/(case+'_worked_prediction.csv'), index=False)
    contributions = pd.DataFrame({'feature':coef.index,'transformed_value':design.iloc[0].to_numpy(),
                                  'coefficient':coef.to_numpy(),'contribution':design.iloc[0].to_numpy()*coef.to_numpy()})
    contributions.to_csv(TABLES/(case+'_contributions.csv'), index=False)
    return {'case':case,'run_id':str(run[0]),'rows_reconstructed':len(pred),
            'max_absolute_difference':float(np.max(np.abs(rebuilt-stored))),
            'intercept':provenance['intercept'], 'example':sample.to_dict('records')[0],
            'scaler_std':{name:float(np.sqrt(audit['scaler_variances'][name])) for name in audit['numeric']},
            'raw_unit_coefficients':{name:float(coef['num__'+name]/np.sqrt(audit['scaler_variances'][name])) for name in audit['numeric']}}


def main():
    catalog=Catalog(ROOT/'data')
    house,_=catalog.load('house');adult,_=catalog.load('adult')
    triple=house.set_index('Id').loc[[1,2,3], ['GrLivArea','SalePrice']].reset_index()
    x=triple.GrLivArea.to_numpy(float);y=triple.SalePrice.to_numpy(float)
    xc=x-x.mean();yc=y-y.mean()
    slope=float(xc@yc/(xc@xc));intercept=float(y.mean()-slope*x.mean())
    X=np.column_stack([np.ones(len(x)),x])
    beta=np.array([intercept,slope]);np.testing.assert_allclose(beta,np.linalg.lstsq(X,y,rcond=None)[0])
    np.testing.assert_allclose(X.T@(y-X@beta),np.zeros(2),atol=1e-7)
    triple['x_centered']=xc;triple['y_centered']=yc
    triple['cross_product']=xc*yc;triple['x_centered_squared']=xc**2
    triple['fitted_for_teaching']=X@beta;triple['residual']=y-X@beta
    triple.to_csv(TABLES/'ols_three_rows.csv',index=False)
    retriever=Retriever(catalog,'house')
    question='CollgCr 社区房屋的平均成交价是多少？'
    hits=retriever.search(question)
    save_json(OUT/'worked_retrieval.json',{'question':question,'documents_in_index':len(retriever.docs),'hits':hits})
    plan=Plan(dataset='house',operation='aggregate',columns=['SalePrice'],
              filters=[{'column':'Neighborhood','op':'eq','value':'CollgCr'}],metric='mean')
    result=Engine(catalog,ROOT/'artifacts/teaching').execute(plan,question,hits)
    result['interaction_mode']='manual_parameters_for_teaching'
    save_json(OUT/'worked_aggregate.json',result)
    expected=house.loc[house.Neighborhood.eq('CollgCr'),'SalePrice']
    np.testing.assert_allclose(result['table'][0]['value'],expected.mean())
    sub=house.loc[house.Neighborhood.eq('CollgCr'), ['Id','Neighborhood','SalePrice']]
    sub.to_csv(TABLES/'collgcr_members.csv',index=False)
    mean_case={'n':len(expected),'sum':float(expected.sum()),'mean':float(expected.mean()),
               'full_n':len(house),'full_mean':float(house.SalePrice.mean()),
               'lotfrontage_n_valid':int(house.LotFrontage.notna().sum()),'lotfrontage_mean':float(house.LotFrontage.mean())}
    # 三行样本仅说明计算，不送入最低要求为100行的交互训练器。
    ols={'ids':[1,2,3],'xbar':float(x.mean()),'ybar':float(y.mean()),'sxy':float(xc@yc),
         'sxx':float(xc@xc),'slope':slope,'intercept':intercept,'sse':float(((y-X@beta)**2).sum())}
    # 数值差分只用于验证文中梯度的代数；这些是确定性代数输入，不是课程数据。
    Z=np.array([[1.,-.8,.3],[1.,.2,-1.1],[1.,1.4,.6],[1.,-.2,.9]])
    yy=np.array([0.,1.,1.,0.]);w=np.array([.3,-.2,.5]);D=np.diag([0.,1.,1.]);C=1.
    def loss(ww):
        eta=Z@ww
        return float((np.logaddexp(0,eta)-yy*eta).mean()+(ww@D@ww)/(2*C*len(yy)))
    def grad(ww):
        return Z.T@(expit(Z@ww)-yy)/len(yy)+(D@ww)/(C*len(yy))
    prob=expit(Z@w)
    hessian=(Z.T*(prob*(1-prob)))@Z/len(yy)+D/(C*len(yy))
    g_numeric=approx_derivative(lambda ww:np.array([loss(ww)]),w).ravel()
    h_numeric=approx_derivative(grad,w)
    np.testing.assert_allclose(grad(w),g_numeric,atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose(hessian,h_numeric,atol=1e-8,rtol=1e-7)
    rhouse=reconstruct('house_primary_linear',house)
    radult=reconstruct('adult_primary_logistic',adult,True)
    proof={'mean_example':mean_case,'ols_three_rows':ols,'reconstruction':[rhouse,radult],
           'logistic_derivative_check':{'gradient_max_abs_error':float(np.max(np.abs(grad(w)-g_numeric))),
             'hessian_max_abs_error':float(np.max(np.abs(hessian-h_numeric))),
             'minimum_hessian_eigenvalue':float(np.linalg.eigvalsh(hessian).min())},
           'source_sha256':{'train_csv':hashlib.sha256((ROOT/'data/incoming/train.csv').read_bytes()).hexdigest()},
           'new_baseline_fits':0,'llm_called':False}
    save_json(OUT/'teaching_calculations.json',proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
