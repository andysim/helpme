add_custom_target(update_version ALL
                  COMMAND ${PYTHON_EXECUTABLE} versioner.py --metaout ${CMAKE_CURRENT_BINARY_DIR}/metadata.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Generating version info")

#add_custom_target(PythonTest
#    COMMAND ${PYTHON_EXECUTABLE} setup.py test
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#    DEPENDS helpmelib
#    COMMENT "Testing Python module"
#)
#add_custom_target(PythonInstall
#    COMMAND ${PYTHON_EXECUTABLE} setup.py install
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#    DEPENDS PythonTest
#    COMMENT "Installing Python module"
#)

include_directories(../src)
pybind11_add_module(helpmecore pywrappers.cc)
target_link_libraries(helpmecore PRIVATE fftw::fftw)

configure_file(../test/fullexample.py ./tests/TestSerial.py COPYONLY)

install(FILES $<TARGET_FILE:helpmecore>
              __init__.py
              ${CMAKE_CURRENT_BINARY_DIR}/metadata.py
        DESTINATION ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/helpme)
