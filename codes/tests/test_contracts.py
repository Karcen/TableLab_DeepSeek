import pytest
from pydantic import ValidationError
from tablelab.contracts import Plan,Filter

@pytest.mark.parametrize('payload',[
 {'dataset':'unknown','operation':'describe'},
 {'dataset':'house','operation':'shell'},
 {'dataset':'house','operation':'describe','code':'print(1)'},
 {'dataset':'house','operation':'train','target':'SalePrice','features':['SalePrice']},
 {'dataset':'house','operation':'train','target':'SalePrice'},
 {'dataset':'house','operation':'train','target':'SalePrice','features':['GrLivArea'],'algorithm':'xgboost'},
 {'dataset':'house','operation':'describe','columns':['Id','Id']},
 {'dataset':'house','operation':'unsupported'},
 {'dataset':'house','operation':'predict','model_id':'../../secret','record':{'x':1}},
 {'dataset':'house','operation':'correlation','columns':['SalePrice']},
 {'dataset':'house','operation':'plot','chart':'scatter','columns':['SalePrice']},
 {'dataset':'house','operation':'aggregate','metric':'mean','columns':[]},
 {'dataset':'house','operation':'describe','limit':0},
 {'dataset':'house','operation':'describe','filters':[{'column':'Id','op':'gt','value':float('inf')}]}
])
def test_reject_invalid_plan(payload):
    with pytest.raises(ValidationError):Plan.model_validate(payload)

@pytest.mark.parametrize('payload',[
 {'column':'age','op':'in','value':30},
 {'column':'age','op':'gt','value':[20,30]},
 {'column':'age','op':'eq'},
 {'column':'age','op':'is_missing','value':1},
 {'column':'age','op':'or','value':30}
])
def test_reject_invalid_filter(payload):
    with pytest.raises(ValidationError):Filter.model_validate(payload)


@pytest.mark.parametrize('payload',[
 {'dataset':'house','operation':'aggregate','metric':'count','columns':['Id','SalePrice']},
 {'dataset':'house','operation':'train','target':'SalePrice','features':['GrLivArea'],'columns':['OverallQual']},
])
def test_ambiguous_extra_measure_rejected(payload):
    with pytest.raises(ValueError):Plan.model_validate(payload)

def test_bar_row_count_has_no_measure():
    p=Plan(dataset='house',operation='plot',chart='bar',metric='count',group_by=['Neighborhood'])
    assert p.columns==[]
