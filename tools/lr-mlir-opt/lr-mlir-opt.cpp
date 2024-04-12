#include "../../../mlir/IR/MLIRContext.h"
#include "../../../mlir/InitAllDialects.h"
#include "../../../mlir/InitAllPasses.h"
#include "../../../mlir/Support/FileUtilities.h"
#include "../../../mlir/Tools/mlir-opt/MlirOptMain.h"

//#include "../../include/LifeRange/Passes.h"

int main(int argc, char **argv) 
{
    //mlir::liferange::registerPasses();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Life Range Pass Driver", registry));
}