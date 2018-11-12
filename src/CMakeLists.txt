set(sources_list
        helpme.cc
        fftw_wrapper.h
        )

include_directories(${FFTW_INCLUDES})
add_library(helpmestatic ${sources_list})
target_link_libraries(helpmestatic PUBLIC ${FFTW_LIBRARIES})
target_link_libraries(helpmestatic PUBLIC OpenMP::OpenMP_CXX)

if(Fortran_ENABLED AND CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    #Enable call to for_rtl_init_() which is required if using the
    #Intel fortran compiler
    target_compile_definitions(helpmestatic PRIVATE -DINTEL_Fortran_ENABLED)
endif()

set_target_properties(helpmestatic PROPERTIES OUTPUT_NAME helpme
                                              EXPORT_NAME helpme)

set(PN ${PROJECT_NAME})

install(DIRECTORY .
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
        FILES_MATCHING PATTERN "*.h")

install(TARGETS helpmestatic
        EXPORT "${PN}Targets"
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

