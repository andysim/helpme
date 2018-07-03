add_custom_target(update_version ALL
                  COMMAND ${PYTHON_EXECUTABLE} versioner.py --metaout ${CMAKE_CURRENT_BINARY_DIR}/metadata.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Generating version info")

add_custom_target(PythonTest
    COMMAND ${PYTHON_EXECUTABLE} setup.py test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS helpmelib
    COMMENT "Testing Python module"
)
add_custom_target(PythonInstall
    COMMAND ${PYTHON_EXECUTABLE} setup.py install
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS PythonTest
    COMMENT "Installing Python module"
)

include_directories(../src)
include_directories(${FFTW_INCLUDES})
link_libraries(${FFTW_LIBRARIES})
pybind11_add_module(helpmelib pywrappers.cc)

configure_file(setup.py . COPYONLY)
configure_file(../LICENSE . COPYONLY)
configure_file(Manifest.in . COPYONLY)
configure_file(../test/fullexample.py ./tests/TestSerial.py COPYONLY)
