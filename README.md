improved_glmnet
===============
This is a software library meant to be a well-written, performance-competitive
competitor to the library <tt>glmnet</tt>. Currently supported features
include:

* solving the elastic net optimization problem
* generating a regularization path
* selecting a lambda value that has excellent generalization performance on the elastic net problem

I have written one usage example per described feature, which you can see in the
<tt>src</tt> directory:
* <tt>cvx_validation.cc</tt>
* <tt>cross_validation.cc</tt>
* <tt>regularization_path.cc</tt> .

For more details, see the preliminary paper I wrote at:
https://www.dropbox.com/s/lyy0mgz8pjpdy38/final.pdf .

###Dependencies
* <tt>g++</tt> (at least v. 4.8).
* <tt>armadillo</tt> (at least v. 4.300).
* a linear algebra subroutine package supported by <tt>armadillo</tt> such as
  <tt>OpenBLAS</tt>.

###Installation
To compile the examples, perform the below steps:

1. You'll need modify the Makefile in the <tt>src</tt> directory to specify a
linear algebra library to link against. I assume you took my advice and are
using <tt>OpenBLAS</tt>. In this case, just change the <tt>OpenBLAS_DIR</tt>
variable to point to the installation directory on your machine. That's it.

2. Run ```make```.

