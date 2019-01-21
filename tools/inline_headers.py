# BEGINLICENSE
#
# This file is part of helPME, which is distributed under the BSD 3-clause license,
# as described in the LICENSE file in the top level directory of this project.
#
# Author: Andrew C. Simmonett
#
# ENDLICENSE

#
# This utility serves as a stripped-down version of the C++ preprocessor, inlining the
# header files pulled in by helpme.h, and working recursively to ensure sub-dependencies
# are also included.  Files are only included once, in the order they're needed.
#

import io
import os
import re

includere = re.compile(r'\s*#\s*include\s*[<"](.*)[">]\s*')
headerguardre = re.compile(r'\s*#((define)|(ifndef)) _HELPME_(.*)_H_')

warning = u"""
//
// WARNING! This file is automatically generated from the sources in the src directory.
// Do not modify this source code directly as any changes will be overwritten
//
"""

def canonicalize_include(name):
    """ Look to see if this include exists in the helPME source """
    if os.path.isfile('../src/' + name):
        return "../src/" + name
    else:
        return None

def cleanup_file(name):
    """ Sanitize included files before inlining them """
    output = []
    for line in io.open(name, encoding="utf-8").readlines():
        # Look for header guards and relabel them to make LGTM's analyzer happy
        match = headerguardre.match(line)
        if match:
            incname = ' _HELPME_STANDALONE_' + match.groups()[3] + '_H_\n'
            line = '#' + match.group(1) + incname
        output.append(line)
    return output

#
# Make the partially inlined version of the header, which still needs Eigen
#
output_array = ["// original file: ../src/helpme.h\n\n"]
output_array.extend(cleanup_file('../src/helpme.h'))

offset = 0
changes_made = True
already_added = []
while changes_made:
    changes_made = False
    for rel_line_number,line in enumerate(output_array[offset:]):
        abs_line_number = offset + rel_line_number
        incmatch = includere.match(line)
        if incmatch:
            helpme_filename = canonicalize_include(incmatch.group(1))
            if helpme_filename:
                if helpme_filename in already_added:
                    # Just comment out this include; it's already been included
                    output_array[abs_line_number] = "// " + line
                else:
                    # We need to include this header
                    replacement_text = ["// original file: %s\n\n" % helpme_filename]
                    replacement_text.extend(cleanup_file(helpme_filename))
                    output_array[abs_line_number:abs_line_number+1] = replacement_text
                    already_added.append(helpme_filename)
                changes_made = True
                offset = abs_line_number
                break

thistext = warning + '\n\n' + "".join(output_array)
existingtext = "".join(io.open('../single_include/helpme_standalone.h', 'r', encoding="utf-8").readlines())

if(thistext != existingtext):
    with io.open('../single_include/helpme_standalone.h', 'w', encoding="utf-8") as fp:
        fp.write(thistext)
