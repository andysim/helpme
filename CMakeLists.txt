cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

################  Options: Overview and Not Otherwise Mentioned  ###############
#
#  <<<  CMake build overview  >>>
#
#    >>> cmake -H. -Bobjdir -DCMAKE_INSTALL_PREFIX=/path/to/helpme/install/dir ...
#    ...
#    -- Generating done
#    -- Build files have been written to: /current/dir/objdir
#
#    >>> cd objdir && make -j`getconf _NPROCESSORS_ONLN`
#    >>> make install
#
#  <<<  Compilers and flags  >>>
#
#    - CMAKE_C_COMPILER "C compiler"
#    - CMAKE_C_FLAGS "Additional C flags"
#    - CMAKE_CXX_COMPILER "C++ compiler"
#    - CMAKE_CXX_FLAGS "Additional C++ flags"
#    - CMAKE_Fortran_COMPILER "Fortran compiler (required for some add-ons)"
#    - CMAKE_Fortran_FLAGS "Additional Fortran flags"
#
#  <<<  Detecting dependencies and add-ons  >>>
#
#    - PYTHON_EXECUTABLE "Python interpreter to use (e.g., /path/to/bin/python2.7)"
#    - SPHINX_ROOT "Root directory for Sphinx: 'bin/sphinx-build' (or similar) should be in this dir."
#

project(helpme
        LANGUAGES C CXX)
set(helpme_URL "http://www.github.com/andysim/helpme")
set(helpme_EMAIL "andrew.simmonett@nih.gov")
set(helpme_LICENSE "BSD-3-clause")
set(helpme_DESCRIPTION "helPME: an efficient library for particle mesh Ewald")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set (CMAKE_CXX_STANDARD 11)

#
# Language bindings options
#
include(custom_color_messages)
option(ENABLE_fortran "Enables the building of Fortran bindings (needs Fortran compiler)" ON)
if(ENABLE_fortran)
    enable_language(Fortran)
    set(Fortran_ENABLED ON)
endif()

#
# Build options
#
include(optionsTools)
option_with_print(ENABLE_OPENMP "Enables OpenMP parallelization" ON)
option_with_print(ENABLE_MPI "Enables MPI parallelization" ON)
option_with_flags(ENABLE_CODE_COVERAGE "Enables details on code coverage" OFF
                  "-ftest-coverage -fprofile-arcs -fPIC -O0 -g")
option_with_flags(ENABLE_BOUNDS_CHECK "Enables bounds check in Fortran" OFF
                  "-ftrapuv -check all -fpstkchk" "-fcheck=all" "-fbounds-check -fcheck-array-temporaries")
option_with_flags(ENABLE_ASAN "Enables address sanitizer" OFF
                  "-fsanitize=address" "-fno-omit-frame-pointer")
option_with_flags(ENABLE_MSAN "Enables address sanitizer" OFF
                  "-fsanitize=memory" "-fno-omit-frame-pointer")
option_with_flags(ENABLE_TSAN "Enables thread sanitizer" OFF
                  "-fsanitize=thread" "-fno-omit-frame-pointer -pie")
option_with_flags(ENABLE_UBSAN "Enables undefined behavior sanitizer" OFF
                  "-fsanitize=undefined" "-fno-omit-frame-pointer")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0")

# FFTW
find_package(FFTW REQUIRED)
if(HAVE_FFTWF)
    add_definitions(-DHAVE_FFTWF=1)
endif()
if(HAVE_FFTWD)
    add_definitions(-DHAVE_FFTWD=1)
endif()
if(HAVE_FFTWL)
    add_definitions(-DHAVE_FFTWL=1)
endif()

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP)
endif()

if(OPENMP_FOUND)
    add_definitions(-DHAVE_OPENMP)
    if(DEFINED CMAKE_C_COMPILER_ID)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    endif()
    if(DEFINED CMAKE_CXX_COMPILER_ID)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    if(DEFINED CMAKE_Fortran_COMPILER_ID)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
    endif()
endif()

# MPI
if(ENABLE_MPI)
    find_package(MPI)
endif()

if(MPI_CXX_FOUND)
    message(STATUS "${Cyan}Found MPI: ${MPI_INCLUDE_PATH}${ColourReset}")
    set(HAVE_MPI TRUE)
    add_definitions("-DHAVE_MPI=1")
else()
    if(ENABLE_MPI)
        message(STATUS "${Red}MPI not found${ColourReset}")
    else()
        message(STATUS "${Red}MPI not searched for${ColourReset}")
    endif()
endif()

# Find Python
set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5)
find_package(PythonLibsNew 2.7 REQUIRED)
message(STATUS "${Cyan}Found Python ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}${ColourReset}: ${PYTHON_EXECUTABLE} (found version ${PYTHON_VERSION_STRING})")
set(Python_ENABLED ON) # For now, we just always assume this is on
add_subdirectory(external/pybind11)
add_subdirectory(python)

# Handle different C++ libraries
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    set(cpplib c++)
else()
    set(cpplib stdc++)
endif()

# The C++ library is linked explicitly because exception handling on macOS appears to be broken otherwise.
set(EXTERNAL_LIBRARIES ${FFTW_LIBRARIES} ${cpplib})

# Documentation
find_package(Doxygen)
find_package(Sphinx)
find_package(Latex)
if(DOXYGEN_EXECUTABLE)
    if(SPHINX_EXECUTABLE)
        add_subdirectory(docs)
    endif()
endif()
add_subdirectory(src)


#
# Testing
#
enable_testing ()
add_subdirectory (test)

