add_mlir_dialect_library(MLIRLifeRange
        LifeRangePass.cpp

        DEPENDS
        MLIRLiveRangePassIncGen

        LINK_LIBS PUBLIC
        MLIRFuncDialect
)