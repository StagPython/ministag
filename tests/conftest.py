import pytest


@pytest.fixture
def minconf():
    return {"numerical": {"nsteps": 1, "nwrite": 1}}
