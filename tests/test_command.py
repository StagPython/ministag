from contextlib import contextmanager
import os
import subprocess
import toml


@contextmanager
def safecd(newdir):
    olddir = os.getcwd()
    try:
        os.chdir(str(newdir))
        yield newdir
    finally:
        os.chdir(olddir)


def test_bare_command(tmp, minconf):
    with (tmp / 'par.toml').open('w') as parfile:
        toml.dump(minconf, parfile)
    with safecd(tmp):
        subprocess.run('ministag', shell=True)
        odir = tmp / 'output'
        assert odir.is_dir()
        assert (odir / 'par.toml').is_file()
        assert (odir / 'time.h5').is_file()
