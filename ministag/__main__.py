"""Make ministag callable."""

from __future__ import annotations
import pathlib
import shutil
import signal
import sys
import typing

from .config import Config

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
    conf = Config.default_()
    par = pathlib.Path('par.toml')
    if not par.is_file():
        print(f"Parameter file `{par}` not found, creating one for you.",
              "Modify it to your taste and rerun ministag.", sep='\n')
        conf.to_file_(par)
        sys.exit()
    conf.update_from_file_(par)
    from . import solver
    rb2d = solver.RunManager(conf)
    if not rb2d.conf.numerical.restart and conf.inout.outdir.is_dir():
        print('Output directory already exists.',
              'Resuming may lead to loss of data.')
        answer = input('Keep on going anyway (y/N)? ')
        if answer.lower() != 'y':
            sys.exit()
        shutil.rmtree(conf.inout.outdir)
    conf.inout.outdir.mkdir(exist_ok=True)
    rb2d.dump_pars(conf.inout.outdir / 'par.toml')
    rb2d.solve(progress=True)


if __name__ == '__main__':
    main()
