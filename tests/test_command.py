import os
import subprocess
from contextlib import contextmanager

import toml


@contextmanager
def safecd(newdir):
    olddir = os.getcwd()
    try:
        os.chdir(str(newdir))
        yield newdir
    finally:
        os.chdir(olddir)


def test_bare_command(tmp_path, minconf):
    with (tmp_path / "par.toml").open("w") as parfile:
        toml.dump(minconf, parfile)
    with safecd(tmp_path):
        subprocess.run("ministag", shell=True)
        odir = tmp_path / "output"
        assert odir.is_dir()
        assert (odir / "par.toml").is_file()
        assert (odir / "time.h5").is_file()
