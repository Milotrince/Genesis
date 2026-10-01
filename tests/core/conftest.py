import importlib
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def solver_plugin(tmp_path, monkeypatch):
    shutil.copytree(Path(__file__).parent / "fixtures" / "solver_data_plugin", tmp_path / "solver_data_plugin")
    monkeypatch.syspath_prepend(tmp_path)
    return importlib.import_module("solver_data_plugin")
