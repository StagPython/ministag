import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import toml


@contextmanager
def safecd(newdir: Path) -> Iterator[Path]:
    olddir = Path.cwd()
    try:
        os.chdir(newdir)
        yield newdir
    finally:
        os.chdir(olddir)


def test_bare_command(tmp_path: Path, minconf: dict[str, dict[str, int]]) -> None:
    with (tmp_path / "par.toml").open("w") as parfile:
        toml.dump(minconf, parfile)
    with safecd(tmp_path):
        subprocess.run("ministag", shell=True)
        odir = tmp_path / "output"
        assert odir.is_dir()
        assert (odir / "par.toml").is_file()
        assert (odir / "time.h5").is_file()
