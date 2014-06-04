improved_glmnet
===============
This is a software library meant to complete
with the library <tt>glmnet</tt> in solving instances of the elastic net
optimization problem.

For more details, see the preliminary paper I wrote at:
https://www.dropbox.com/s/lyy0mgz8pjpdy38/final.pdf

###Dependencies
* <tt>g++</tt> (at least v. 4.8).
* <tt>armadillo</tt> (at least v. 4.300).
* a linear algebra subroutine package supported by <tt>armadillo</tt> such as
  <tt>OpenBLAS</tt>.

###Installation
I have written the following usage examples, which you can see in the
<tt>src</tt> directory:
```c++
1. cvx_validation.cc
2. cross_validation.cc
3. regularization_path.cc
```
To compile these examples, perform the below steps:

1. You'll need modify the Makefile in the <tt>src</tt> directory to specify a
linear algebra library to link against. I assume you took my advice and are
using <tt>OpenBLAS</tt>. In this case, just change <tt>OpenBLAS_DIR</tt>
variable to point to the installation directory on your machine. That's it.

2. Run ```make```.

