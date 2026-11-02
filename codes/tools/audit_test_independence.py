"""Audit saved row usage and test-set overlap without fitting or scoring models."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,zipfile
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.catalog import Catalog,SPECS
from tablelab.engine import predictor_fingerprints

ROOT=Path(__file__).resolve().parents[2]
HOLDOUT_COMPLETE=(ROOT/'artifacts/holdout/adult/complete.json').exists()
OUT=ROOT/'codes/verification/research'/('independence_audit_after_holdout' if HOLDOUT_COMPLETE else 'independence_audit')
DATASETS=['house','adult','nba','students']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 frames={ds:Catalog(ROOT/'data').load(ds)[0].set_index('_row_id',drop=False) for ds in DATASETS}
 before={str((ROOT/'data/processed'/f'{ds}.csv').relative_to(ROOT)):sha(ROOT/'data/processed'/f'{ds}.csv') for ds in DATASETS}
 adult=frames['adult'];full_cols=[c for c in adult if not c.startswith('_') and c not in ['income','income_gt_50k']]
 assert len(full_cols)==14
 full_fp=predictor_fingerprints(adult[full_cols])
 selected_fp=predictor_fingerprints(adult[list(SPECS['adult'].defaults)])
 checks=[];coverage=[]
 for ds,df in frames.items():
  train_all=set();val_all=set();val_counts=Counter();folds=[]
  for folder in sorted((ROOT/'artifacts/research'/ds).glob('fold*')):
   assert (folder/'complete.json').exists(),folder
   split=pd.read_csv(folder/'split.csv',dtype={'row_id':str});train=set(split.loc[split.partition=='train','row_id']);val=set(split.loc[split.partition=='validation','row_id'])
   assert not (train&val) and train|val<=set(df.index)
   train_all|=train;val_all|=val;val_counts.update(val)
   row={'dataset':ds,'fold':int(folder.name[4:]),'n_train':len(train),'n_validation':len(val),'train_validation_row_overlap':len(train&val),'inner_folds':0,'inner_outside_outer_train':0,'inner_row_overlap':0,'group_overlap':None,'inner_group_overlap':None,'chronology_ok':None,'inner_chronology_ok':None}
   if ds=='adult':
    row['group_overlap']=len(set(full_fp.loc[list(train)])&set(full_fp.loc[list(val)]));row['inner_group_overlap']=0
    assert row['group_overlap']==0
    assert all(':train:' in rid for rid in train|val)
   if ds=='nba':
    row.update({'train_season_min':df.loc[list(train),'season'].min(),'train_season_max':df.loc[list(train),'season'].max(),'validation_season_min':df.loc[list(val),'season'].min(),'validation_season_max':df.loc[list(val),'season'].max()})
    row['chronology_ok']=row['train_season_max']<row['validation_season_min'];row['inner_chronology_ok']=True;assert row['chronology_ok']
   inner=json.loads((folder/'inner_splits.json').read_text());assert len(inner)==3
   for item in inner:
    a=set(item['train_row_ids']);b=set(item['validation_row_ids']);row['inner_folds']+=1
    row['inner_outside_outer_train']+=len((a|b)-train);row['inner_row_overlap']+=len(a&b)
    assert a|b<=train and not a&b
    if ds=='adult':
     overlap=len(set(full_fp.loc[list(a)])&set(full_fp.loc[list(b)]));row['inner_group_overlap']+=overlap;assert overlap==0
    if ds=='nba':
     ordered=df.loc[list(a),'season'].max()<df.loc[list(b),'season'].min();row['inner_chronology_ok']&=bool(ordered);assert ordered
   folds.append(row);checks.append(row)
  union=train_all|val_all;unseen=set(df.index)-union
  coverage.append({'dataset':ds,'available_rows':len(df),'outer_train_unique':len(train_all),'outer_validation_unique':len(val_all),'used_in_outer_any_role':len(union),'outer_training_only':len(train_all-val_all),'outer_validation_only':len(val_all-train_all),'outside_outer_experiment':len(unseen),'maximum_validation_repeats':max(val_counts.values()),'minimum_validation_repeats_among_validated':min(val_counts.values()),'missing_target_rows':int(df[SPECS[ds].targets[0]].isna().sum())})
  pd.DataFrame({'row_id':df.index,'outer_training':[rid in train_all for rid in df.index],'outer_validation':[rid in val_all for rid in df.index],'outer_validation_count':[val_counts[rid] for rid in df.index]}).to_csv(OUT/f'{ds}_row_coverage.csv',index=False)
 # Include earlier benchmark, demonstration and saved API-check model runs.
 saved={'train':{ds:set() for ds in DATASETS},'validation':{ds:set() for ds in DATASETS},'other':{ds:set() for ds in DATASETS}}
 evidence=[];unknown=[];seen=set()
 for base in [ROOT/'artifacts',ROOT/'models',ROOT/'codes/artifacts']:
  if not base.exists():continue
  candidates=list(base.rglob('split.csv'))+list(base.rglob('*predictions*.csv'))
  for path in sorted(candidates):
   resolved=path.resolve()
   if resolved in seen:continue
   seen.add(resolved);cols=pd.read_csv(path,nrows=0).columns
   if 'row_id' not in cols:continue
   selected=['row_id']+(['partition'] if 'partition' in cols else [])
   part=pd.read_csv(path,usecols=selected,dtype={'row_id':str})
   if 'partition' not in part:part['partition']='validation'
   active=part.loc[~part.partition.isin(['unused','excluded','original_test'])]
   roles=active.partition.map(lambda r:'train' if r=='train' else 'validation' if r in ['validation','interactive_validation','test'] else 'other')
   for role in roles.unique():
    ids=set(active.loc[roles==role,'row_id'].dropna())
    for ds in DATASETS:
     ds_ids={rid for rid in ids if rid.startswith(ds+':')};saved[role][ds].update(ds_ids);unknown.extend([{'file':str(path.relative_to(ROOT)),'row_id':rid} for rid in ds_ids-set(frames[ds].index)])
   evidence.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'records_in_file':len(part),'active_records':len(active),'inactive_records':len(part)-len(active),'adult_original_test_active_rows':int(active.row_id.str.startswith('adult:original_test:').sum())})
 assert not unknown,unknown[:3]
 for row in coverage:
  ds=row['dataset'];union=set().union(*(saved[role][ds] for role in saved))
  row['used_in_all_saved_splits_or_predictions']=len(union);row['outside_all_saved_splits_or_predictions']=len(set(frames[ds].index)-union)
 train_ids=set(adult.loc[adult._source_split=='train'].index);test_ids=set(adult.loc[adult._source_split=='original_test'].index)
 tr=full_fp.loc[list(train_ids)];te=full_fp.loc[list(test_ids)];common=set(tr)&set(te);overlap_test={rid for rid in test_ids if full_fp.loc[rid] in common};overlap_train={rid for rid in train_ids if full_fp.loc[rid] in common}
 model_common=set(selected_fp.loc[list(train_ids)])&set(selected_fp.loc[list(test_ids)])
 candidate=test_ids-overlap_test
 flags=pd.DataFrame({'row_id':sorted(test_ids)})
 flags['full_predictor_fingerprint']=flags.row_id.map(full_fp)
 flags['overlaps_train_full_predictors']=flags.row_id.isin(overlap_test)
 flags['repeated_group_within_test']=flags.full_predictor_fingerprint.isin(set(te[te.duplicated(keep=False)]))
 flags['candidate_after_cross_file_exclusion']=flags.row_id.isin(candidate)
 flags.to_csv(OUT/'adult_test_overlap_flags.csv',index=False)
 histories={role:len(saved[role]['adult']&test_ids) for role in saved}
 if HOLDOUT_COMPLETE:
  assert histories=={'train':0,'validation':len(candidate),'other':0},histories
  assert saved['validation']['adult']&test_ids==candidate
 else:
  assert sum(histories.values())==0,histories
 # Reconcile the hash-based counts with a direct raw-column merge.
 direct=adult.loc[list(test_ids),['_row_id']+full_cols].merge(adult.loc[list(train_ids),full_cols].drop_duplicates(),on=full_cols,how='left',indicator=True,validate='many_to_one')
 direct_overlap=set(direct.loc[direct['_merge']=='both','_row_id'])
 assert direct_overlap==overlap_test
 six_cols=list(SPECS['adult'].defaults)
 direct_six=adult.loc[list(test_ids),six_cols].merge(adult.loc[list(train_ids),six_cols].drop_duplicates(),on=six_cols,how='left',indicator=True,validate='many_to_one')
 assert int((direct_six['_merge']=='both').sum())==int(selected_fp.loc[list(test_ids)].isin(model_common).sum())
 assert len(adult.loc[list(test_ids),full_cols])-len(adult.loc[list(test_ids),full_cols].drop_duplicates())==int(te.duplicated().sum())
 adult_details={'direct_merge_reconciled':True,'six_field_direct_merge_reconciled':True,'training_rows':len(train_ids),'original_test_rows':len(test_ids),'fingerprint_columns':full_cols,'shared_full_predictor_groups':len(common),'affected_training_rows':len(overlap_train),'affected_test_rows':len(overlap_test),'test_rows_after_cross_file_exclusion':len(candidate),'unique_candidate_groups':int(full_fp.loc[list(candidate)].nunique()),'training_duplicate_extra_rows':int(tr.duplicated().sum()),'test_duplicate_extra_rows':int(te.duplicated().sum()),'test_rows_in_repeated_groups':int(te.duplicated(keep=False).sum()),'candidate_duplicate_extra_rows':len(candidate)-int(full_fp.loc[list(candidate)].nunique()),'shared_model_input_six_field_groups':len(model_common),'test_rows_matching_training_on_six_model_fields':int(selected_fp.loc[list(test_ids)].isin(model_common).sum()),'prior_use_in_saved_models':histories,'test_file_previously_profiled':True,'score_computed_in_this_audit':False,'targets_used_to_select_candidate_rows':False}
 inventory=[]
 for path in sorted((ROOT/'data/raw').glob('*/source.zip')):
  with zipfile.ZipFile(path) as archive:
   inventory.append({'archive':str(path.relative_to(ROOT)),'sha256':sha(path),'members':[n for n in archive.namelist() if not n.endswith('/')]})
 nba=frames['nba'];nba_info={'season_count':int(nba.season.nunique()),'first_season':nba.season.min(),'last_season':nba.season.max(),'nonunique_name_season_rows':int(nba.duplicated(['player_name','season'],keep=False).sum()),'nonunique_name_season_keys':int(nba.loc[nba.duplicated(['player_name','season'],keep=False),['player_name','season']].drop_duplicates().shape[0])}
 for name,digest in before.items():assert sha(ROOT/name)==digest
 pd.DataFrame(checks).to_csv(OUT/'nested_split_checks.csv',index=False)
 pd.DataFrame(coverage).to_csv(OUT/'coverage_summary.csv',index=False)
 pd.DataFrame(evidence).to_csv(OUT/'saved_run_evidence.csv',index=False)
 report={'audit_date':'2026-11-02','scope':'Saved data, split row IDs, inner split row IDs and prediction row IDs; no model fitting or test scoring.','datasets':coverage,'adult':adult_details,'nba':nba_info,'outer_folds_checked':len(checks),'inner_folds_checked':sum(r['inner_folds'] for r in checks),'saved_files_checked':len(evidence),'active_adult_test_occurrences':sum(x['adult_original_test_active_rows'] for x in evidence),'archive_inventory':inventory,'data_alias':{'codes_data_is_symlink':(ROOT/'codes/data').is_symlink(),'codes_data_resolves_to_data':(ROOT/'codes/data').resolve()==(ROOT/'data').resolve()},'processed_source_sha256':before,'data_unchanged':True,'limits':'Absence from saved model records does not prove labels were never viewed; original Adult test was previously inspected for data quality. Six-field model-input matches do not identify individual people.'}
 (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({'coverage':coverage,'adult':adult_details,'nba':nba_info,'outer_folds_checked':len(checks),'inner_folds_checked':sum(r['inner_folds'] for r in checks),'saved_files_checked':len(evidence)},ensure_ascii=False))

if __name__=='__main__':main()
