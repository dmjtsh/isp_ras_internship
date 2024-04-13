get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS
        ${dialect_libs}
        MLIROptLib
)

add_llvm_executable(lr-mlir-opt lr-mlir-opt.cpp)
target_link_libraries(lr-mlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(lr-mlir-opt)