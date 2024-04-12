#ifndef MLIR_DIALECT_AFFINE_PASSES
#define MLIR_DIALECT_AFFINE_PASSES

include "../mlir/Pass/PassBase.td"

def LifeRange : Pass<"life-range", "func::FuncOp"> 
{
    let summary = "Life Range Pass to show life cycles of variables";
    
    let constructor = "mlir::liferange::createLifeRangePass()";
    
    let dependentDialects = ["memref::MemRefDialect"];
}

#endif 