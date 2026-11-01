"""确定性合成夹具，只验证软件。不得作为四个原始数据集使用。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from tablelab.engine import write_json


def make_fixture_root(root):
    root=Path(root);(root/'processed').mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261030);n=360
    house=pd.DataFrame({'Id':np.arange(1,n+1),'OverallQual':rng.integers(2,10,n),
        'GrLivArea':rng.normal(1600,360,n).round(1),'YearBuilt':rng.integers(1950,2010,n),
        'TotalBsmtSF':rng.normal(800,160,n).round(1),'GarageCars':rng.integers(0,4,n),
        'Neighborhood':rng.choice(['TEST_A','TEST_B','TEST_C'],n),'YrSold':np.tile(np.arange(2006,2012),n//6)})
    house['SalePrice']=50000+house['GrLivArea']*60+house['OverallQual']*8000+rng.normal(0,15000,n)
    house.loc[[4,19,63],'GrLivArea']=np.nan
    adult=pd.DataFrame({'age':rng.integers(18,70,n),'education-num':rng.integers(6,17,n),
        'hours-per-week':rng.integers(15,65,n),'workclass':rng.choice(['Private','State-gov'],n),
        'occupation':rng.choice(['Tech-support','Sales','Prof-specialty'],n),'marital-status':rng.choice(['Never-married','Married-civ-spouse'],n),
        'sex':rng.choice(['Female','Male'],n),'fnlwgt':rng.integers(10000,300000,n)})
    probability=1/(1+np.exp(-(-6+.045*adult['age']+.24*adult['education-num']+.03*adult['hours-per-week'])))
    adult['income_gt_50k']=(rng.random(n)<probability).astype(int)
    adult['income']=np.where(adult['income_gt_50k']==1,'>50K','<=50K')
    adult['education']='TEST_EDU_'+adult['education-num'].astype(str)
    adult['_source_split']=np.where(np.arange(n)<300,'train','original_test')
    p=30;s=12;m=p*s
    nba=pd.DataFrame({'player_name':[f'TEST_P{x:03d}' for x in np.repeat(np.arange(p),s)],
        'season':np.tile([f'{year}-{(year+1)%100:02d}' for year in range(2010,2010+s)],p),
        'age':rng.integers(19,36,m),'player_height':rng.normal(199,9,m),'player_weight':rng.normal(96,10,m),
        'draft_year':rng.choice(['2006','2008','2010','Undrafted'],m),'team_abbreviation':rng.choice(['TEST_X','TEST_Y'],m),
        'gp':rng.integers(20,83,m),'reb':rng.uniform(1,12,m),'ast':rng.uniform(.5,10,m)})
    nba['pts']=np.maximum(0,12-.15*(nba['age']-25)+.04*(nba['player_height']-199)+rng.normal(0,3,m))
    students=pd.DataFrame({'gender':rng.choice(['female','male'],n),'race/ethnicity':rng.choice(['group A','group B','group C','group D','group E'],n),
        'parental level of education':rng.choice(["bachelor's degree",'high school',"master's degree"],n),
        'lunch':rng.choice(['standard','free/reduced'],n),'test preparation course':rng.choice(['completed','none'],n)})
    students['math score']=np.clip(64+8*students['test preparation course'].eq('completed')+rng.normal(0,12,n),0,100).round()
    students['reading score']=np.clip(students['math score']+rng.normal(4,8,n),0,100).round()
    students['writing score']=np.clip(students['reading score']+rng.normal(-2,6,n),0,100).round()
    manifest={'format_version':1,'warning':'SYNTHETIC SOFTWARE FIXTURES; NOT ORIGINAL ASSIGNMENT DATA','datasets':{}}
    for name,df in {'house':house,'adult':adult,'nba':nba,'students':students}.items():
        df['_row_id']=[f'SYNTHETIC:{name}:{i:06d}' for i in range(len(df))]
        dest=root/'processed'/f'{name}.csv';df.to_csv(dest,index=False)
        manifest['datasets'][name]={'source_kind':'synthetic_test','rows':len(df),'columns':list(df),
            'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source_page':'tests/fixtures.py',
            'generator_seed':20261030,'license_note':'自建合成测试夹具；不是原始数据。'}
    write_json(root/'manifest.json',manifest)
    return root

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='verification/synthetic_data')
    args=parser.parse_args();make_fixture_root(args.out)
    print('已生成明确标注的合成夹具，仅供软件验证。')
