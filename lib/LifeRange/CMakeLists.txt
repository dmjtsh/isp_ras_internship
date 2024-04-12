add_mlir_dialect_library(MLIRLifeRange
		LifeRangePass.cpp
		
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/LifeRange

        LINK_LIBS PUBLIC
        MLIRFuncDialect
)