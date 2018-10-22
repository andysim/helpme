# Add any new tests to this list or the one below!
set( SOURCES_UNITTESTS_TESTS
    unittest-cartesiantransform.cpp
    unittest-coulombkappasweep.cpp
    unittest-dispersionkappasweep.cpp
    unittest-fft.cpp
    unittest-fullrun.cpp
    unittest-fullrun-multipoles.cpp
    unittest-gammafunction.cpp
    unittest-gridsize.cpp
    unittest-lattice.cpp
    unittest-latticeupdates.cpp
    unittest-matrix.cpp
    unittest-powers.cpp
    unittest-potential.cpp
    unittest-splines.cpp
    unittest-string.cpp
)
if(HAVE_MPI)
    set( SOURCES_UNITTESTS_PARALLEL_TESTS
        unittest-coulomb-rec-parallel.cpp
        unittest-dispersion-rec-parallel.cpp
    )
endif()



#
# Stuff below shouldn't need to be modified
#

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${FFTW_INCLUDES})

set( SOURCES_UNITTESTS_MAIN unittest-main.cpp )

# A function to check that the given hard-coded list contains all headers + sources in the given folder.
function(CheckFileList LIST_VAR FOLDER)
  set(MESSAGE " should be added to the variable ${LIST_VAR}")
  set(MESSAGE "${MESSAGE} in ${CMAKE_CURRENT_LIST_FILE}\n")
  file(GLOB GLOBBED_LIST "${FOLDER}/*.cpp"
                         "${FOLDER}/*.hpp")
  list(REMOVE_ITEM GLOBBED_LIST ${${LIST_VAR}})
  foreach(EXTRA_ITEM ${GLOBBED_LIST})
    string(REPLACE "${CATCH_DIR}/" "" RELATIVE_FILE_NAME "${EXTRA_ITEM}")
    message(AUTHOR_WARNING "The file \"${RELATIVE_FILE_NAME}\"${MESSAGE}")
  endforeach()
endfunction()

set( UNITTEST_DIR ${PROJECT_SOURCE_DIR}/test/unittests )
set( HEADER_DIR ${PROJECT_SOURCE_DIR}/external/Catch2/single_include )

# Check if all sources are listed; warn if not.
set( SOURCES_ALL
    ${SOURCES_UNITTESTS_MAIN}
    ${SOURCES_UNITTESTS_TESTS}
    ${SOURCES_UNITTESTS_PARALLEL_TESTS}
)
foreach( name ${SOURCES_ALL} )
    list( APPEND SOURCES_ALL_PATH ${UNITTEST_DIR}/${name} )
endforeach()
CheckFileList( SOURCES_ALL_PATH ${UNITTEST_DIR} )

# Create serial target names.
string( REPLACE ".cpp" "" BASENAMES_UNITTESTS_TESTS "${SOURCES_UNITTESTS_TESTS}" )
set( TARGETS_UNITTESTS_TESTS ${BASENAMES_UNITTESTS_TESTS} )

# Create parallel target names.
string( REPLACE ".cpp" "" BASENAMES_UNITTESTS_PARALLEL_TESTS "${SOURCES_UNITTESTS_PARALLEL_TESTS}" )
set( TARGETS_UNITTESTS_PARALLEL_TESTS ${BASENAMES_UNITTESTS_PARALLEL_TESTS} )

set( TARGETS_ALL ${TARGETS_UNITTESTS_TESTS} ${TARGETS_UNITTESTS_PARALLEL_TESTS} CatchMain )

# Define program targets:
add_library( CatchMain OBJECT ${UNITTEST_DIR}/${SOURCES_UNITTESTS_MAIN} ${HEADER_DIR}/catch.hpp )

foreach( name ${TARGETS_UNITTESTS_PARALLEL_TESTS} )
    add_executable( ${name} ${name}.cpp $<TARGET_OBJECTS:CatchMain> ${HEADER_DIR}/catch.hpp )
    set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
    target_link_libraries( ${name} PRIVATE ${EXTERNAL_LIBRARIES} ${MPI_CXX_LIBRARIES})
    target_compile_definitions(${name} PRIVATE "-DHAVE_MPI=1")
    target_include_directories(${name} PRIVATE "${MPI_CXX_INCLUDE_PATH}")
    add_test(NAME ${name}
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./${name} ${MPIEXEC_POSTFLAGS}
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/unittests)
endforeach()

foreach( name ${TARGETS_UNITTESTS_TESTS} )
    add_executable( ${name} ${name}.cpp $<TARGET_OBJECTS:CatchMain> ${HEADER_DIR}/catch.hpp )
    target_link_libraries( ${name} PRIVATE ${EXTERNAL_LIBRARIES})
    add_test(NAME ${name}
             COMMAND ${name} 
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/unittests)
endforeach()

foreach( name ${TARGETS_ALL} )
    target_include_directories( ${name} PRIVATE ${HEADER_DIR} )

    set_property(TARGET ${name} PROPERTY CXX_STANDARD 11)

    # GCC specific warning go here
    if ( CMAKE_CXX_COMPILER_ID MATCHES "GCC" )
        # Actually keep these
        target_compile_options( ${name}  PRIVATE -Wweak-vtables -Wexit-time-destructors -Wglobal-constructors -Wmissing-noreturn -Wno-int-in-bool-context)
    endif()
    # Clang specific warning go here
    if ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
        # Actually keep these
        target_compile_options( ${name}  PRIVATE -Wweak-vtables -Wexit-time-destructors -Wglobal-constructors)
    endif()
    if ( CMAKE_CXX_COMPILER_ID MATCHES "MSVC" )
        target_compile_options( ${name}  PRIVATE /W4 /w44265 /WX )
    endif()
endforeach()
