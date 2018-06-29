#!/bin/bash
# BEGINLICENSE
#
# This file is part of helPME, which is distributed under the BSD 3-clause license,
# as described in the LICENSE file in the top level directory of this project.
#
# Author: Andrew C. Simmonett
#
# ENDLICENSE


# This script will check a predefined list of C++ source and header files, using clang-format.
# By default, the clang-format in the path will be used.  To use a version from a different
# location, simply set the environmental variable CLANG_FORMAT_DIR before calling this, e.g.
#
# export CLANG_FORMAT_DIR=/usr/local/bin/
#
# with the trailing forward slash present.  The return code is set to zero if no errors are
# found, else 1, so this script can be used in CI setups to automatically check formatting.


CLANGFORMAT=${CLANG_FORMAT_DIR}clang-format

# Make sure we're in the top level directory
cd `dirname $0`
cd ..

declare -a extensions=("*.h" "*.cc" "*.cpp" "*.hpp")
declare -a directories=("src" "test" "test/unittests")

# Make a temporary file, and ensure it's nuked even if interrupted
tmpfile=$(mktemp)
trap 'rm -f -- "$tmpfile"' INT TERM HUP EXIT
shopt -s nullglob

echo "Checking C++ file formatting..."
returncode=0
for dir in "${directories[@]}"
do
    for ext in "${extensions[@]}"
    do
        for file in ${dir}/${ext}
        do
            $CLANGFORMAT --style=file $file > "$tmpfile"
            diff $file "$tmpfile" > /dev/null
            if [ $? -ne 0 ]
            then
                returncode=1
                echo
                echo "****************************************************"
                echo "Formatting problem detected.  Run"
                echo
                echo "${CLANGFORMAT} --style=file -i $file"
                echo
                echo "from the top directory, or apply the following diff:"
                echo "----------------------------------------------------"
                diff $file "$tmpfile"
                echo "****************************************************"
                echo
                echo
            fi
        done
    done
done

if [ $returncode -eq 0 ]
then
    echo "C++ file formats are good!"
fi

exit $returncode
