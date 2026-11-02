"""Export fitted predictions/outer metrics at float64 precision, without refits.

Preserve native outputs, model hashes and search decisions. NumPy scalar
promotion can otherwise retain float32 through regression inverse scaling,
and pandas' float32 CSV formatter need not preserve float64 metric arithmetic.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,shutil,argparse
from datetime import datetime,timezone
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.research.expanded import sha,model_predict,get_metric
from tablelab.research.experiment import dump
from expanded_paths import local_path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'


def main(partial):
    threadpool_limits(1)
    specification={'code_sha256':sha(Path(__file__)),'rule':'Cast model predictions to float64 before inverse target scaling; export all decimal digits; recompute metrics in float64',
                   'training':'No fitting, parameter changes, threshold changes or search decisions',
                   'preservation':'Native precision predictions, metrics and completion marker archived per flow; model SHA256 unchanged',
                   'persistence':'Use exact last-season points from saved sequence master, rather than decoding a float32 scaled value'}
    protocol=DATA/'precision_protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())['specification']==specification
    else:dump(protocol,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'specification':specification})
    protocol_hash=sha(protocol)
    jobs=json.loads((DATA/'jobs.json').read_text())
    seq=pd.read_csv(DATA/'nba_sequence/sequence_master.csv').set_index('_row_id')
    count=0;max_difference=0.
    for t in jobs:
        folder=local_path(t['folder'])/(t['model']+'_'+t['method'])
        complete=folder/'complete.json'
        if not complete.exists():
            assert partial
            continue
        done=json.loads(complete.read_text())
        if 'precision_protocol_sha256' in done:
            assert done['precision_protocol_sha256']==protocol_hash
            count+=1
            continue
        o=joblib.load(local_path(t['cache_path']))['outer']
        if t['method']=='fixed':
            if t['model']=='Mean':
                p=np.full(len(o['actual']),float(np.mean(o['y'],dtype=np.float64))*o['scale']+o['centre'],dtype=np.float64)
                train_p=np.full(len(o['y']),p[0],dtype=np.float64)
            else:
                p=seq.loc[o['validation_ids'],'lag1_pts'].to_numpy(dtype=np.float64)
                train_p=seq.loc[o['train_ids'],'lag1_pts'].to_numpy(dtype=np.float64)
        else:
            assert sha(folder/'model.joblib')==done['model_sha256']
            model=joblib.load(folder/'model.joblib')
            p=np.asarray(model_predict(model,o['V'],t['cls'],t['model']),dtype=np.float64)*o['scale']+o['centre']
            train_p=np.asarray(model_predict(model,o['X'],t['cls'],t['model']),dtype=np.float64)*o['scale']+o['centre']
        native=pd.read_csv(folder/'predictions.csv').set_index('row_id')
        old=native.loc[o['validation_ids'],'predicted'].to_numpy()
        difference=float(np.max(np.abs(old-p)))
        # This postprocessing is allowed to correct rounding, not prediction changes.
        assert difference <= max(.1,1e-6*float(np.max(np.abs(p)))),(t,difference)
        max_difference=max(max_difference,difference)
        archive=folder/'native_precision';archive.mkdir(exist_ok=True)
        for name in ['metrics.csv','predictions.csv','complete.json']:
            if not (archive/name).exists():shutil.copy2(folder/name,archive/name)
        scores=pd.read_csv(folder/'metrics.csv')
        for part,y,pp in [('validation',o['actual'],p),('train',o['y']*o['scale']+o['centre'],train_p)]:
            vals=get_metric(np.asarray(y,dtype=np.float64),pp,t['cls'])
            for key,value in vals.items():scores.loc[scores.partition==part,key]=value
        scores.to_csv(folder/'metrics.csv',index=False,float_format='%.17g')
        pd.DataFrame({'row_id':o['validation_ids'],'actual':o['actual'],'predicted':p}).to_csv(folder/'predictions.csv',index=False,float_format='%.17g')
        done.update({'precision_protocol_sha256':protocol_hash,'native_predictions_sha256':sha(archive/'predictions.csv'),
                     'max_prediction_rounding_correction':difference,'prediction_sha256':sha(folder/'predictions.csv'),
                     'metrics_sha256':sha(folder/'metrics.csv')})
        dump(complete,done)
        count+=1
        if count%100==0:print(f'Float64 export {count}/{len(jobs)}',flush=True)
    dump(DATA/('precision_partial.json' if partial else 'precision_complete.json'),
         {'processed_jobs':count,'expected_jobs':len(jobs),'precision_protocol_sha256':protocol_hash,
          'max_correction_in_this_run':max_difference,'partial':partial})
    print(json.dumps({'processed':count,'expected':len(jobs),'max_rounding_correction':max_difference,'partial':partial}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');main(parser.parse_args().partial)
