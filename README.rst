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

Just type::

    $ ministag

and the code will run with simple parameters for a short time. Results can be
found in a directory named ``output``. Also stored is the an input parameter
file corresponding to this run and named ``par.toml``. To run other cases,
simply copy that file elsewhere, modify it as needed and run ``ministag``
again.
