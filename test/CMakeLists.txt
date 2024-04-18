configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LIFE_RANGE_TEST_DEPENDS
        FileCheck count not
        lr-mlir-opt
)

add_lit_testsuite(check-life-range "Running the Life Range tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${LIFE_RANGE_TEST_DEPENDS}
)


set_target_properties(check-life-range PROPERTIES FOLDER "Tests")

add_lit_testsuites(LIFE_RANGE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${LIFE_RANGE_TEST_DEPENDS})
