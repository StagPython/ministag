import pytest


@pytest.fixture
def minconf() -> dict[str, dict[str, int]]:
    return {"numerical": {"nsteps": 1, "nwrite": 1}}
