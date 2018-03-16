# Add any new tests to this list!
set(SOURCES_SPHINX
    bibliography.rst
    index.rst
    introduction.rst
)

#
# Doxygen
#
set(DOXYGEN_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/doxygen")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
file(MAKE_DIRECTORY "${DOXYGEN_OUTPUT_DIR}")

configure_file(
    ${DOXYGEN_INPUT_DIR}/Doxyfile.in
    ${DOXYGEN_OUTPUT_DIR}/Doxyfile
    @ONLY
)

add_custom_command(
    OUTPUT "${DOXYGEN_OUTPUT_DIR}/xml/index.xml"
    COMMAND "${DOXYGEN_EXECUTABLE}"
    DEPENDS "${DOXYGEN_OUTPUT_DIR}/Doxyfile"
    WORKING_DIRECTORY "${DOXYGEN_OUTPUT_DIR}"
    COMMENT "Generating Doxygen API docs..."
)

add_custom_target(DoxygenDocs
                  DEPENDS "${DOXYGEN_OUTPUT_DIR}/xml/index.xml")

#
# Sphinx
#


set(SPHINX_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sphinx")
set(SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx")

foreach(SOURCE_FILE ${SOURCES_SPHINX})
    list (APPEND SPHINX_INPUT_SOURCES "${SPHINX_INPUT_DIR}/${SOURCE_FILE}")
endforeach()

configure_file(
    ${SPHINX_INPUT_DIR}/conf.py.in
    ${SPHINX_OUTPUT_DIR}/conf.py
    @ONLY
)

file(MAKE_DIRECTORY "${SPHINX_OUTPUT_DIR}/html")
add_custom_command(OUTPUT "${SPHINX_OUTPUT_DIR}/html"
                   COMMAND ${SPHINX_EXECUTABLE}
                           -q -b html
                           -c "${SPHINX_OUTPUT_DIR}"
                           "${SPHINX_INPUT_DIR}"
                           "${SPHINX_OUTPUT_DIR}/html"
                   DEPENDS ${SPHINX_INPUT_SOURCES}
                   DEPENDS "${DOXYGEN_OUTPUT_DIR}/xml/index.xml")


add_custom_target(SphinxHtml
                  DEPENDS "${SPHINX_OUTPUT_DIR}/html"
                  COMMENT "Building HTML documentation with Sphinx")

# This will trigger all doxygen and sphinx builds
add_custom_target(docs
                  DEPENDS "${SPHINX_OUTPUT_DIR}/html")
