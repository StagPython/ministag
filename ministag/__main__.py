"""Make ministag callable."""

import importlib
import pathlib
import signal
import sys


def _sigint_handler(*_):
    """Handler of SIGINT signal."""
    print('\nYour will to stop me is staggering.')
    sys.exit()


def main():
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
    rb2d = solver.RayleighBenardStokes(par if par.is_file() else None)
    rb2d.dump_pars(par)
    rb2d.solve()


if __name__ == '__main__':
    main()
