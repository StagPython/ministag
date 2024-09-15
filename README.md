MiniStag
--------

2D convection solver.

This is inspired from the 2D convection code by Paul Tackley
available at http://jupiter.ethz.ch/~pjt/Convection2D.m

Installation
============

`ministag` is coded in Python and can be installed using [`uv` to manage Python
environments](https://docs.astral.sh/uv/).

You can then install `ministag` with the following commands from the root of
the repository (i.e. where the `pyproject.toml` file is):

```sh
uv tool install .
```

This installs `ministag` in a dedicated virtual environment and make it
available in a standard location (usually `~/.local/bin`).

Running
=======

`ministag` is configurable through a parameter file named `par.toml`.  If
that file doesn't exist, run ministag once to create it with default values:

```sh
ministag
```

You can then modify `par.toml` as desired and rerun `ministag` to perform
your simulation.  You can run the following to display a short explanation of
the available options:

```sh
ministag help
```
