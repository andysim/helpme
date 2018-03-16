#!/bin/bash
# BEGINLICENSE
#
# This file is part of helPME, which is distributed under the BSD 3-clause license,
# as described in the LICENSE file in the top level directory of this project.
#
# Author: Andrew C. Simmonett
#
# ENDLICENSE

#
# This is a cheesy helper script to build the code coverage map.
#

# Make sure we're in the build subdir of the top level directory

if [ -z "$PYTHON" ]; then
    PYTHON=`which python`
fi

if ! [ -x "$(command -v cmake)" ]; then
  echo 'Error: cmake is not available' >&2
  exit 1
fi

cd `dirname $0`/..
TOPDIR=${PWD}
BUILDDIR=$TOPDIR/build_coverage
mkdir -p $TOPDIR/build_coverage
cd $BUILDDIR

rm -rf *

cmake \
-DPYTHON_EXECUTABLE=$PYTHON \
-DCMAKE_C_COMPILER=gcc \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_Fortran_COMPILER=gfortran \
-DENABLE_CODE_COVERAGE=ON \
..

make -j4
ctest
cd src/CMakeFiles/helpmestatic.dir
echo running
echo
echo $TOPDIR/external/gcovr/scripts/gcovr . -r $TOPDIR/src/ --html-details --html -o coverage.html
echo
echo from ${PWD}
echo
$TOPDIR/external/gcovr/scripts/gcovr . -r $TOPDIR/src/ --html-details --html -o ../../../coverage.html
