import pathlib
import pytest


@pytest.fixture
def tmp(tmpdir):
    return pathlib.Path(str(tmpdir))


@pytest.fixture
def minconf():
    return {'numerical': {'nsteps': 1, 'nwrite': 1}}
