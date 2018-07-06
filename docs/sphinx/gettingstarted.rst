.. # BEGINLICENSE
.. #
.. # This file is part of helPME, which is distributed under the BSD 3-clause license,
.. # as described in the LICENSE file in the top level directory of this project.
.. #
.. # Author: Andrew C. Simmonett
.. #
.. # ENDLICENSE

.. _`sec:gettingstarted`:

===============
Getting Started
===============

Installation
============

General installation
--------------------

|helPME| uses CMake for configuring and building.  It's generally a good idea
to build in a directory other than the source directory, *e.g.*, to build using
4 cores:

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make -j4
    $ make test

On some systems, the default setup detected by CMake may not be what's
appropriate; always check the information provided by CMake to ensure that the
desired compilers / tools are detected.  Here's a much more complete example
that specifies MPI compiler wrappers to use, as well as the installation
directory:

.. code-block:: bash

    $ CC=mpicc CXX=mpicxx FC=mpif90 FFTWROOT=/path/to/fftw/installation cmake .. -DPYTHON_EXECUTABLE=/path/to/python -DCMAKE_INSTALL_PREFIX=/path/to/install/helpme/into
    $ make -j4
    $ make test
    $ make PythonInstall
    $ make docs
    $ make install

Installation may not be necessary, depending on your choice of language and
build setup.  Here's a quick overview of what is needed for each language
choice.  Examples of the library's usage can be found in the :repo:`test`
directory.

C++
---

Because the library is written in C++ this is straightforward.  After building
and installing, simply adding the include directory to the compiler include
path using the ``-I`` flag will allow the library to be used as demonstrated in
:testcase:`fullexample.cpp` for OpenMP parallel or
:testcase:`fullexample_parallel.cpp` for hybrid OpenMP/MPI parallel.  For
convenience, a :repo:`single header <single_include/helpme_standalone.h>` is
available to be simply dropped into a project's source to avoid any building of
|helPME|.

If importing |helPME| headers in this way, some compile-time defines must be
specified by the host project.  If MPI is to be used, ``-DHAVE_MPI=1`` should
be added to the C++ compiler flags.  Moreover, |helPME| uses the FFTW API for
Fourier transforms.  It is the host program's responsibility to make sure the
appropriate headers are added to the compiler include path and that the
appropriate libraries are linked in.  Because FFTW provides different precision
modes, one or more of ``-DHAVE_FFTWF=1``, ``-DHAVE_FFTWD=1`` or
``-DHAVE_FFTWL=1`` should be added to the compile flags to activate single,
double and long double precision modes, respectively.

C
-

To use the library from C there are two possible approaches.  The first is to
follow the `General installation`_ instructions and ensure that the include
directory is in the C compiler's include path, and the |helPME| library from
`lib` is linked.  From there, the library can be used by including the
``helpme.h`` header and calling the API, as demonstrated in
:testcase:`fullexample.c` for OpenMP parallel or
:testcase:`fullexample_parallel.c` for hybrid OpenMP / MPI parallel.

Another approach that elides the build and install step is to include the
:repo:`single header <single_include/helpme_standalone.h>` file into the
project as `helpme.h`. Then import the :repo:`C wrapper <src/helpme.cc>` and
ensure that it is added to the build system.  See the `C++`_ section for
important information about build flags that must be included to activate
different features, if following the single header strategy.

Fortran
-------

The fortran build instructions are very similar to those for C, as the library
uses the ISO C bindings feature of Fortran 90 to wrap the C API into a Fortran
module.  Either of the two approaches outlined for `C`_ builds will work for
Fortran; the only additional requirement is to import the :repo:`Fortran
interface <src/helpme.F90>` and ensure that it is properly build by the host
program.   The Fortran API is demonstrated in :testcase:`fullexample.F90` for
OpenMP parallel or :testcase:`fullexample_parallel.F90` for hybrid OpenMP / MPI
parallel.

Python
------

To install, simply follow the `General installation`_ instructions and make
sure the appropriate Python executable is detected.  After calling ``make`` and
``make test`` to verify the installation, the ``make PythonInstall`` command
will make the library available to Python.  The API is demostrated in
:testcase:`fullexample.py`.
