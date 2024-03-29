set(sources_list
        helpme.cc
        )

include_directories(${FFTW_INCLUDES})
include_directories(${MKL_INCLUDE_DIR})
add_library(helpmestatic STATIC ${sources_list})
target_link_libraries(helpmestatic ${FFTW_LIBRARIES})
add_library(helpme SHARED ${sources_list})
target_link_libraries(helpme ${FFTW_LIBRARIES})

if(Fortran_ENABLED AND CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    #Enable call to for_rtl_init_() which is required if using the
    #Intel fortran compiler
    target_compile_definitions(helpmestatic PRIVATE -DINTEL_Fortran_ENABLED)
endif()

set_target_properties(helpmestatic PROPERTIES OUTPUT_NAME helpme EXPORT_NAME helpme)

# Make sure headers are installed
install(DIRECTORY . DESTINATION include FILES_MATCHING PATTERN "*.h")

install(TARGETS helpmestatic helpme
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
