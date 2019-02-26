# BEGINLICENSE
#
# This file is part of helPME, which is distributed under the BSD 3-clause license,
# as described in the LICENSE file in the top level directory of this project.
#
# Author: Andrew C. Simmonett
#
# ENDLICENSE

"""Dummy setup.py file solely for the purposes of getting an on-the-fly
computed version number into the conda recipe.

"""
import sys
from distutils.core import setup

def version_func():
    import subprocess

    command = 'python python/versioner.py --formatonly --format={version}'
    process = subprocess.Popen(command.split(), shell=False, stdout=subprocess.PIPE)
    (out, err) = process.communicate()
    if sys.version_info >= (3, 0):
        return out.decode('utf-8').strip()
    else:
        return out.strip()

setup(
    version=version_func(),
)
