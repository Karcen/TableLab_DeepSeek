from pathlib import Path
import pytest
from tablelab.catalog import Catalog
from tablelab.engine import Engine
from .fixtures import make_fixture_root

@pytest.fixture
def root(tmp_path):return make_fixture_root(tmp_path/'data')

@pytest.fixture
def catalog(root):return Catalog(root)

@pytest.fixture
def engine(catalog,tmp_path):return Engine(catalog,tmp_path/'runs')
