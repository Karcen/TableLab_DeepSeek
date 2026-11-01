from pathlib import Path
import json
import pytest
from tools.evaluate_llm import require_course_source
from tablelab.catalog import DataError

@pytest.mark.parametrize('source',['original','course_dataset'])
def test_accepted_course_types(source):
    require_course_source({'source_kind':source})

@pytest.mark.parametrize('source',['synthetic_test','unknown',''])
def test_reject_unreviewed_types(source):
    with pytest.raises(DataError): require_course_source({'source_kind':source})

def test_packaged_manifests_pass_acceptance_gate():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'data/manifest.json').read_text())['datasets']
    assert set(manifest)=={'house','adult','nba','students'}
    for entry in manifest.values(): require_course_source(entry)
