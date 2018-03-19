add_subdirectory(unittests)

# Add any new sources here
set(SOURCES_HELPME
    fftw_wrapper.h
    gamma.h
    lapack_wrapper.h
    helpme.h
    matrix.h
    memory.h
    powers.h
    splines.h
)


foreach(SOURCE_FILE ${SOURCES_HELPME})
    list (APPEND HELPME_INPUT_SOURCES "${PROJECT_SOURCE_DIR}/src/${SOURCE_FILE}")
endforeach()

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/single_include)
include_directories(${FFTW_INCLUDES})
link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${FFTW_LIBRARIES})

# Ensure headers are inlined every time a build is triggered
add_custom_command(
    OUTPUT StandaloneHeader
    COMMAND ${PYTHON_EXECUTABLE} inline_headers.py
    DEPENDS ${HELPME_INPUT_SOURCES}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tools"
    COMMENT "Generating single header..."
)

add_custom_target(SingleHeader
                  DEPENDS StandaloneHeader
)

# Find only the static library for linking with C++/C/Fortran wrappers
set(helpmestatic_name ${CMAKE_BINARY_DIR}/src/helpme.a)

# C example
add_executable (RunCWrapper fullexample.c)
add_dependencies(RunCWrapper helpmestatic)
target_link_libraries(RunCWrapper ${helpmestatic_name})
set_target_properties(RunCWrapper PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME CWrapperTest COMMAND RunCWrapper)

# CXX example
add_executable (RunCXXWrapper fullexample.cpp)
add_test(NAME CXXWrapperTest COMMAND RunCXXWrapper)

# CXX example using standalone header file
add_executable (RunCXXWrapperStandalone fullexample.cpp)
add_dependencies(RunCXXWrapperStandalone SingleHeader)
target_compile_definitions(RunCXXWrapperStandalone PRIVATE "-DBUILD_STANDALONE=1")
add_test(NAME CXXWrapperStandaloneTest COMMAND RunCXXWrapperStandalone)

# CXX MPI
if(HAVE_MPI)
    add_executable (RunCXXMPIWrapper fullexample_parallel.cpp)
    set_target_properties(RunCXXMPIWrapper PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
    target_compile_definitions(RunCXXMPIWrapper PRIVATE "-DHAVE_MPI=1")
    target_include_directories(RunCXXMPIWrapper PRIVATE "${MPI_CXX_INCLUDE_PATH}")
    target_link_libraries(RunCXXMPIWrapper ${MPI_CXX_LIBRARIES})
    add_test(NAME CXXMPIWrapperTest
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} RunCXXMPIWrapper ${MPIEXEC_POSTFLAGS})
endif()

# Fortran example
if(Fortran_ENABLED)
    add_executable(RunFortranWrapper ${PROJECT_SOURCE_DIR}/src/helpme.f90 fullexample.f90 )
    add_dependencies(RunFortranWrapper helpmestatic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        set(cpplib c++)
    else()
        set(cpplib stdc++)
    endif()
    target_link_libraries(RunFortranWrapper ${helpmestatic_name} ${cpplib})
    add_test(NAME FortranWrapper COMMAND RunFortranWrapper)
endif()

# Add python tests
if(Python_ENABLED)
    add_test(NAME PythonWrappers
        COMMAND ${PYTHON_EXECUTABLE} setup.py test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../python
    )
endif()
