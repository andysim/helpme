include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
# GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
set(CMAKECONFIG_INSTALL_DIR "share/cmake/helpme")

add_custom_target(update_version ALL
                  COMMAND ${PYTHON_EXECUTABLE} versioner.py --metaout ${CMAKE_CURRENT_BINARY_DIR}/metadata.py
                                                            --cmakeout ${CMAKE_CURRENT_BINARY_DIR}/metadata.cmake
                  COMMAND ${CMAKE_COMMAND} -DWTO="${CMAKE_CURRENT_BINARY_DIR}"
                                           -DPN="helpme"
                                           -P ${CMAKE_CURRENT_BINARY_DIR}/metadata.cmake
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Generating version info")

# <<<  Build  >>>

include_directories(../src)
pybind11_add_module(pymod pywrappers.cc)
target_link_libraries(pymod PRIVATE fftw::fftw)

configure_file(../test/fullexample.py ./tests/TestSerial.py COPYONLY)

set(PN helpme)
install(TARGETS pymod
        EXPORT "${PN}Targets"
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/helpme)
install(EXPORT "${PN}Targets"
        NAMESPACE "${PN}::"
        DESTINATION ${CMAKECONFIG_INSTALL_DIR})

# <<<  Install  >>>

install(FILES #$<TARGET_FILE:pymod>
              __init__.py
              ${CMAKE_CURRENT_BINARY_DIR}/metadata.py
              ${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/helpme)

# <<<  Export Config  >>>

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/${PN}Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PN}Config.cmake
        INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PN}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PN}ConfigVersion.cmake
        DESTINATION ${CMAKECONFIG_INSTALL_DIR})
