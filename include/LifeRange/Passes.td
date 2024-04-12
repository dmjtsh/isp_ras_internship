#ifndef LIFERANGE_PASS
#define LIFERANGE_PASS

include "../../../mlir/Pass/PassBase.td"

def LifeRange : Pass<"life-range", "func::FuncOp"> 
{
    let summary = "Life Range Pass to show life cycles of variables";
    
    //let constructor = "mlir::liferange::createLifeRangePass()";
    
    let dependentDialects = ["memref::MemRefDialect"];
}

#endif 