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

file(MAKE_DIRECTORY "${SPHINX_OUTPUT_DIR}/latex")
add_custom_command(OUTPUT "${SPHINX_OUTPUT_DIR}/latex"
                   COMMAND ${SPHINX_EXECUTABLE}
                           -q -b latex
                           -c "${SPHINX_OUTPUT_DIR}"
                           "${SPHINX_INPUT_DIR}"
                           "${SPHINX_OUTPUT_DIR}/latex"
                   DEPENDS ${SPHINX_INPUT_SOURCES}
                   DEPENDS "${DOXYGEN_OUTPUT_DIR}/xml/index.xml")
add_custom_target(SphinxLatex
                  DEPENDS "${SPHINX_OUTPUT_DIR}/latex"
                  COMMENT "Building LaTeX documentation with Sphinx")

# Make sure headers are installed
install(DIRECTORY "${SPHINX_OUTPUT_DIR}/html" DESTINATION doc)
install(DIRECTORY "${SPHINX_OUTPUT_DIR}/latex" DESTINATION doc)


if(LATEX_FOUND)
    add_custom_command(OUTPUT "${SPHINX_OUTPUT_DIR}/latex/helPME.pdf"
                       COMMAND make
                       WORKING_DIRECTORY "${SPHINX_OUTPUT_DIR}/latex"
                       DEPENDS ${SPHINX_OUTPUT_DIR}/latex)
    add_custom_target(SphinxPdf
                      DEPENDS "${SPHINX_OUTPUT_DIR}/latex"
                      COMMENT "Building PDF documentation with Sphinx")
    install(FILES "${SPHINX_OUTPUT_DIR}/latex/helPME.pdf" DESTINATION doc)
    add_custom_target(docs
                      DEPENDS "${SPHINX_OUTPUT_DIR}/latex"
                      DEPENDS "${SPHINX_OUTPUT_DIR}/latex/helPME.pdf"
                      DEPENDS "${SPHINX_OUTPUT_DIR}/html")
else()
    add_custom_target(docs
                      DEPENDS "${SPHINX_OUTPUT_DIR}/latex"
                      DEPENDS "${SPHINX_OUTPUT_DIR}/html")
endif()
