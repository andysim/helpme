set(PN ${PROJECT_NAME})

set(sources_list helpme.cc)

add_library(helpmelib ${sources_list})
target_link_libraries(helpmelib PUBLIC fftw::fftw)
target_link_libraries(helpmelib PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(helpmelib INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                                               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PN}>)
set_target_properties(helpmelib PROPERTIES OUTPUT_NAME helpme
                                           EXPORT_NAME helpme)

if(Fortran_ENABLED AND CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    #Enable call to for_rtl_init_() which is required if using the
    #Intel fortran compiler
    target_compile_definitions(helpmelib PRIVATE -DINTEL_Fortran_ENABLED)
endif()

set(interface_sources_list
    cartesiantransform.h
    fftw_wrapper.h
    gamma.h
    gridsize.h
    helpme.h
    lapack_wrapper.h
    matrix.h
    memory.h
    mpi_wrapper.h
    powers.h
    splines.h
    string_utils.h
)

add_custom_target(SingleHeader ALL
    COMMAND ${PYTHON_EXECUTABLE} inline_headers.py
    DEPENDS ${interface_sources_list}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tools"
    COMMENT "Generating single header...")


add_library(intf_C INTERFACE)
target_link_libraries(intf_C INTERFACE helpmelib)
#set_target_properties(RunCWrapper PROPERTIES LINKER_LANGUAGE CXX)

add_library(intf_CXX INTERFACE)
target_include_directories(intf_CXX INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                                              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/single_include>
                                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PN}>)
target_link_libraries(intf_CXX INTERFACE fftw::fftw)
target_link_libraries(intf_CXX INTERFACE OpenMP::OpenMP_CXX)

add_library(intf_Fortran INTERFACE)
target_sources(intf_Fortran INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/helpme.F90>
                                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PN}/helpme.F90>)
target_link_libraries(intf_Fortran INTERFACE helpmelib ${cpplib})
#set_target_properties(intf_Fortran PROPERTIES LINKER_LANGUAGE Fortran)

install(DIRECTORY .
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
        FILES_MATCHING PATTERN "*.h")
install(FILES ${PROJECT_SOURCE_DIR}/single_include/helpme_standalone.h
              ${PROJECT_SOURCE_DIR}/src/helpme.F90
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN})

install(TARGETS helpmelib intf_C intf_CXX intf_Fortran
        EXPORT "${PN}Targets"
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
