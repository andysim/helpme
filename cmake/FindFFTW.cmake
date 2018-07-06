# - Find the FFTW library
#
# Usage:
#   find_package(FFTW [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   FFTW_FOUND               ... true if fftw is found on the system
#   FFTW_LIBRARIES           ... full path to fftw library
#   FFTW_INCLUDES            ... fftw include directory
#
# The following variables will be checked by the function
#   FFTW_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW_LIBRARY            ... fftw library to use
#   FFTW_INCLUDE_DIR        ... fftw include directory
#

#If environment variable FFTWDIR is specified, it has same effect as FFTW_ROOT
if( NOT FFTW_ROOT AND EXISTS "$ENV{FFTW_ROOT}")
  set( FFTW_ROOT $ENV{FFTW_ROOT} )
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT FFTW_ROOT )
  pkg_check_modules( PKG_FFTW QUIET "fftw3" )
endif()

set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX}  ${CMAKE_SHARED_LIBRARY_SUFFIX} )

if( FFTW_ROOT )
message( "Searching for FFTW in " ${FFTW_ROOT})
  #find libs
  find_library(
    FFTW_LIB
    NAMES "fftw3"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTWF_LIB
    NAMES "fftw3f"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTWL_LIB
    NAMES "fftw3l"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )

  #find includes
  find_path(
    FFTW_INCLUDES
    NAMES "fftw3.h"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
  )

else()

  find_library(
    FFTW_LIB
    NAMES "fftw3"
    PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  )

  find_library(
    FFTWF_LIB
    NAMES "fftw3f"
    PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  )

  find_library(
    FFTWL_LIB
    NAMES "fftw3l"
    PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  )

  find_path(
    FFTW_INCLUDES
    NAMES "fftw3.h"
    PATHS ${PKG_FFTW_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
  )

endif( FFTW_ROOT )

set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )

if(FFTWF_LIB)
    message(STATUS "${Cyan}Found single precision FFTW: ${FFTWF_LIB}${ColourReset}")
    set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTWF_LIB})
    set(HAVE_FFTWF TRUE)
else()
    message(STATUS "${Red}Single precision FFTW not found${ColourReset}")
endif()

if(FFTW_LIB)
    message(STATUS "${Cyan}Found double precision FFTW: ${FFTW_LIB}${ColourReset}")
    set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_LIB})
    set(HAVE_FFTWD TRUE)
else()
    message(STATUS "${Red}Double precision FFTW not found${ColourReset}")
endif()

if(FFTWL_LIB)
    message(STATUS "${Cyan}Found long double precision FFTW: ${FFTWL_LIB}${ColourReset}")
    set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTWL_LIB})
    set(HAVE_FFTWL TRUE)
else()
    message(STATUS "${Red}Long double precision FFTW not found${ColourReset}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
                                  FFTW_INCLUDES FFTW_LIBRARIES)
mark_as_advanced(FFTW_INCLUDES FFTW_LIBRARIES FFTW_LIB FFTWF_LIB FFTWL_LIB)
