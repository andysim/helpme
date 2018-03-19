<p align="center">
<a href="https://travis-ci.org/andysim/helpme"> <img src="https://travis-ci.org/andysim/helpme.svg?branch=master" /></a>
<a href="https://codecov.io/gh/andysim/helpme"> <img src="https://img.shields.io/codecov/c/github/andysim/helpme/master.svg" /></a>
<a href="https://opensource.org/licenses/BSD-3-Clause"><img src=https://img.shields.io/github/license/andysim/helpme.svg /></a>
</p>

# About #

**h**elPME: an **e**fficient **l**ibrary for **p**article **m**esh **E**wald.
The recursive acronym is a tip of the hat to early open source software tools
and reflects the recursive algorithms that are key to helPME's support for
arbitrary operators. The library is freely available and is designed to be easy
to use, with minimal setup code required.

## Features ##

* Available as a single C++ header.
* Support for C++/C/Fortran/Python bindings.
* Arbitrary operators including *r*<sup>-1</sup> (Coulomb) and *r*<sup>-6</sup>
  (dispersion).
* Ability to use any floating point precision mode, selectable at run time.
* Three dimensional parallel decomposition with MPI.
* OpenMP parallel threading within each MPI instance (still a work in
  progress).
* Support for arbitrary triclinic lattices and orientations thereof.
* Arbitrary order multipoles supported (still a work in progress).
* Memory for coordinates and forces is taken directly from the caller's pool,
  avoiding copies.

## License ##

helPME is distributed under the
[BSD-3-clause](https://opensource.org/licenses/BSD-3-Clause) open source
license, as described in the LICENSE file in the top level of the repository.
Some external dependencies are used that are licensed under different terms, as
enumerated below.

## Dependencies ##
* Either [FFTW](http://www.fftw.org/)
  [(GPL license)](https://opensource.org/licenses/gpl-license) or
  [MKL](https://software.intel.com/en-us/mkl)
  [(ISSL license)](https://software.intel.com/en-us/license/intel-simplified-software-license)
  required to carry out fast Fourier transforms.
* [CMake](https://cmake.org) required if building the code
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [pybind11](https://github.com/pybind/pybind11) required if Python bindings
  are to be built [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [Catch2](https://github.com/catchorg/Catch2) for unit testing 
  [(BSL license)](https://opensource.org/licenses/BSL-1.0).

## Requirements ##
helPME is written in C++11, and should work with any modern (well, non-ancient)
C++ compiler.  Python and Fortran bindings are optional, and are built by
default.

## Authors ##
Andrew C. Simmonett (NIH)
Lori A. Burns (GA Tech)
Daniel R. Roe (NIH)
Bernard R. Brooks (NIH)
