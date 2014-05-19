improved_glmnet
===============

Dependencies:
* g++ (C++11 compatible)
* armadillo (at least v. 4.300)

The next thing you'll need to do is modify the Makefile to compile armadillo against a lineary algebra library. Here are some examples of edits you should make:

* OpenBLAS: this is what I've been using. Change OpenBLAS_DIR in the Makefile to point to the installation directory on your machine, and that's it.
* Intel MKL: ???

