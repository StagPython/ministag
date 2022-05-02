MiniStag
--------

2D convection solver.

This is inspired from the 2D convection code by Paul Tackley
available at http://jupiter.ethz.ch/~pjt/Convection2D.m

Installation
============

``ministag`` is coded in Python and can be installed using ``pip``.  Make sure
your version of ``pip`` is up-to-date::

    $ python3 -m pip install -U --user pip

You can then install ``ministag`` with the following commands from the root of
the repository (i.e. where the ``pyproject.toml`` file is).

You might want to create a virtual environment first (skip these to install in
your regular user directory, often ``~/.local/bin``)::

    $ python3 -m venv venv_ministag
    $ source venv_ministag/bin/activate

You can then install with the following::

    $ python3 -m pip install -U .

Running
=======

``ministag`` is configurable through a parameter file named ``par.toml``.  If
that file doesn't exist, run ministag once to create it with default values::

    $ ministag

You can then modify ``par.toml`` as desired and rerun ``ministag`` to perform
your simulation.  You can run the following to display a short explanation of
the available options::

    $ ministag help
