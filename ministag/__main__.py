"""Make ministag callable."""

import importlib
import signal
import sys


def _sigint_handler(*_):
    """Handler of SIGINT signal."""
    print('\nYour will to stop me is staggering.')
    sys.exit()


def main():
    """Entry point."""
    signal.signal(signal.SIGINT, _sigint_handler)
    solver = importlib.import_module('ministag.solver')
    rb2d = solver.RayleighBenardStokes('par.toml')
    rb2d.solve()


if __name__ == '__main__':
    main()
