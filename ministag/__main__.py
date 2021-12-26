"""Make ministag callable."""

from __future__ import annotations
import importlib
import pathlib
import shutil
import signal
import sys
import typing

if typing.TYPE_CHECKING:
    from typing import Any, NoReturn


def _sigint_handler(*_: Any) -> NoReturn:
    """Handler of SIGINT signal."""
    print('\nYour will to stop me is staggering.')
    sys.exit()


def main() -> None:
    """Entry point."""
    signal.signal(signal.SIGINT, _sigint_handler)
    print(r"""
        {_}
        /=-'
  )____//
 _//---\|_
/         /
""")
    solver = importlib.import_module('ministag.solver')
    par = pathlib.Path('par.toml')
    rb2d = solver.RayleighBenardStokes(parfile=par if par.is_file() else None)
    if not rb2d.conf.numerical.restart and rb2d.outdir.is_dir():
        print('Output directory already exists.',
              'Resuming may lead to loss of data.')
        answer = input('Keep on going anyway (y/N)? ')
        if answer.lower() != 'y':
            sys.exit()
        shutil.rmtree(rb2d.outdir)
    rb2d.outdir.mkdir(exist_ok=True)
    rb2d.dump_pars(rb2d.outdir / 'par.toml')
    rb2d.solve(progress=True)


if __name__ == '__main__':
    main()
