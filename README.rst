MiniStag
--------

2D convection solver.

This is inspired from the 2D convection code by Paul Tackley
available at http://jupiter.ethz.ch/~pjt/Convection2D.m

=================
Installing
=================

You can install the executable to be run anywhere. If you are within
a python virtual environment, just type:
``python3 -m pip install -U .``
This installs dependencies and the ministag executable in
``/path/to/env/bin``.

Otherwise, type:
``python3 -m pip install -U --user .``
to install the ministag excutable in ``~/.local/bin``, which you need
to add in your path.

=================
Running
=================

Just type:
``ministag``
and the code will run with simple parameters for a short
time. Results are stored in a directory named ``output`` in the form
of npz, hdf5 and pdf files. Also stored is the an input parameter file
corresponding to this run and named ``par.toml``. To run other cases,
simply copy that file elsewhere, modify it as needed and run
``ministag`` again. 
