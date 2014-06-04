improved_glmnet
===============
This is a software library meant to be a well-written, performance-competitive
alternative to the library <tt>glmnet</tt>. Currently supported features
include:

* solving the elastic net optimization problem
* generating a regularization path
* selecting a lambda value that has excellent generalization performance on the
  elastic net problem

I have written one usage example per described feature, which you can see in the
<tt>src</tt> directory:

* <tt>cvx_validation.cc</tt>
* <tt>cross_validation.cc</tt>
* <tt>regularization_path.cc</tt> .

For more details, see [the preliminary paper about the library](https://www.dropbox.com/s/lyy0mgz8pjpdy38/final.pdf).

###Dependencies
* <tt>g++</tt> (at least v. 4.8).
* <tt>armadillo</tt> (at least v. 4.300).
* a linear algebra subroutine package supported by <tt>armadillo</tt> such as
  <tt>OpenBLAS</tt>.

###Installation
To compile the examples and the rest of the <tt>improved_glmnet</tt> source
code, perform the below steps:

1. You'll need modify the Makefile in the <tt>src</tt> directory to specify a
linear algebra library to link against. I assume you took my advice and are
using <tt>OpenBLAS</tt>. In this case, just change the <tt>OpenBLAS_DIR</tt>
variable to point to the installation directory on your machine. That's it.

2. Run ```make```.

###Usage
Each example program expects paths to the data set you'll using. There
are several sample datasets of different sizes included in the <tt>tests</tt>
directory that you can use. So, after you have built the examples, you can run
<tt>cvx_validation.cc</tt> with the below command.

```
cvx_validation.out 0.5 ../tests/fat/_100_300/_A ../tests/fat/_100_300/_b ../tests/fat/_100_300/_z
```

See the comments in any of the example source files for more detailed usage instructions.
