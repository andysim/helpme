add_subdirectory(unittests)

if(HAVE_MPI)
    link_libraries(${MPI_CXX_LIBRARIES})
endif()

# C example
#set_target_properties(RunCWrapper PROPERTIES LINKER_LANGUAGE CXX)
add_executable(RunCWrapper fullexample.c)
target_link_libraries(RunCWrapper intf_C)
add_test(NAME CWrapperTest COMMAND RunCWrapper)

# CXX DHFR benchmark example
add_executable (DHFRBenchmark dhfr_benchmark.cpp)
target_link_libraries(DHFRBenchmark intf_CXX)
configure_file(data/dhfr_charges.txt . COPYONLY)
configure_file(data/dhfr_c6s.txt . COPYONLY)
configure_file(data/dhfr_coords.txt . COPYONLY)

# CXX example
add_executable (RunCXXWrapper fullexample.cpp)
target_link_libraries(RunCXXWrapper intf_CXX)
add_test(NAME CXXWrapperTest COMMAND RunCXXWrapper)

# CXX example using standalone header file
add_executable (RunCXXWrapperStandalone fullexample.cpp)
target_link_libraries (RunCXXWrapperStandalone intf_CXX)
#add_dependencies(RunCXXWrapperStandalone SingleHeader)
#target_link_libraries(RunCXXWrapperStandalone ${EXTERNAL_LIBRARIES})
target_compile_definitions(RunCXXWrapperStandalone PRIVATE "BUILD_STANDALONE=1")
add_test(NAME CXXWrapperStandaloneTest COMMAND RunCXXWrapperStandalone)

# CXX MPI
if(HAVE_MPI)
    # CXX example
    add_executable (RunCXXMPIWrapper fullexample_parallel.cpp)
    target_link_libraries(RunCXXMPIWrapper ${EXTERNAL_LIBRARIES})
    add_test(NAME CXXMPIWrapperTestX
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 2 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CXXMPIWrapperTestY
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 2 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CXXMPIWrapperTestZ
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 1 2 ${MPIEXEC_POSTFLAGS})

    # C example
    add_executable (RunCMPIWrapper fullexample_parallel.c)
    target_link_libraries(RunCMPIWrapper helpmelib)
    set_target_properties(RunCMPIWrapper PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME CMPIWrapperTestX
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 2 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CMPIWrapperTestY
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 2 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CMPIWrapperTestZ
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 1 2 ${MPIEXEC_POSTFLAGS})

    # Fortran example
    if(Fortran_ENABLED)
        link_libraries(${MPI_Fortran_LIBRARIES})
        add_executable(RunFortranMPIWrapper ${PROJECT_SOURCE_DIR}/src/helpme.F90 fullexample_parallel.F90 )
        set_target_properties(RunFortranMPIWrapper PROPERTIES LINKER_LANGUAGE Fortran)
        target_link_libraries(RunFortranMPIWrapper helpmelib ${cpplib})
        add_test(NAME FortranMPIWrapperTestX
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 2 1 1 ${MPIEXEC_POSTFLAGS})
        add_test(NAME FortranMPIWrapperTestY
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 1 2 1 ${MPIEXEC_POSTFLAGS})
        add_test(NAME FortranMPIWrapperTestZ
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 1 1 2 ${MPIEXEC_POSTFLAGS})
    endif()
endif()

# Fortran example
if(Fortran_ENABLED)
    add_executable(RunFortranWrapper fullexample.F90)
    target_link_libraries(RunFortranWrapper intf_Fortran)
    add_test(NAME FortranWrapper COMMAND RunFortranWrapper)
    #target_link_libraries(RunFortranWrapper helpmelib ${cpplib})
    #set_target_properties(RunFortranWrapper PROPERTIES LINKER_LANGUAGE Fortran)
endif()

# Add python tests
if(${INSTALL_PYMOD})
    add_test(NAME PythonWrappers
             COMMAND ${PYTHON_EXECUTABLE} -m pytest -rws -v --capture=sys ${CMAKE_CURRENT_SOURCE_DIR}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../python
    )
    install(FILES test_helpme.py
            DESTINATION ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/helpme/tests)
endif()


# Make sure examples are installed
set(EXAMPLEFILES fullexample.c fullexample.cpp fullexample.F90 fullexample.py)
if(HAVE_MPI)
    set(EXAMPLEFILES ${EXAMPLEFILES} fullexample_parallel.c fullexample_parallel.cpp fullexample_parallel.F90)
endif()
install(FILES ${EXAMPLEFILES}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/helpme/examples)
