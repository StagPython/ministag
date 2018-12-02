import pathlib
import pytest


@pytest.fixture
def tmp(tmpdir):
    return pathlib.Path(str(tmpdir))
