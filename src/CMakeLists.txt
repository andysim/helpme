set(sources_list
        helpme.cc
        )

include_directories(${FFTW_INCLUDES})
add_library(helpmestatic STATIC ${sources_list})

if(Fortran_ENABLED AND CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    #Enable call to for_rtl_init_() which is required if using the
    #Intel fortran compiler
    target_compile_definitions(helpmestatic PRIVATE -DINTEL_Fortran_ENABLED)
endif()

set_target_properties(helpmestatic PROPERTIES PREFIX ""
                                              OUTPUT_NAME helpme
                                              EXPORT_NAME helpme
                                              SUFFIX ".a")
if(HAVE_MPI)
    set_target_properties(helpmestatic PROPERTIES LINKER_LANGUAGE "CXX" COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
    target_compile_definitions(helpmestatic PRIVATE "-DHAVE_MPI=1")
    target_include_directories(helpmestatic PRIVATE "${MPI_CXX_INCLUDE_PATH}")
    target_link_libraries(helpmestatic ${MPI_CXX_LIBRARIES})
endif()
