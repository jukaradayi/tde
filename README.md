[![Build Status](https://travis-ci.org/jukaradayi/tde.svg?branch=master)](https://travis-ci.org/jukaradayi/tde)
===============================
Term Discovery Evaluation
===============================

DESCRIPTION

* Free software: GPLv3 license

INSTALLATION

To install the tde package, first you need to create a conda environment with
the right packages::

  $ conda create --name TDE --file conda-requirements.txt
  $ source activate TDE
  $ pip install -r pip-requirements.txt
  $ python setup.py build
  $ python setup.py develop

Note that the packages makes use of cython, so the installation will call GCC.

FREEZING

To build a frozen version::

  $ python setup.py build_ext --inplace
  $ python setup_freeze.py build_exe
  $ python move_build.py CORPUS OUTPUTDIR
  
USAGE 

To evaluate your pairs, use the eval_track2.py script in the tde folder. 
To get a complete doc of how to use the script::

  $ python eval_track2.py --help

ISSUES

Feel free to raise an issue if you encounter any problem with the package.
