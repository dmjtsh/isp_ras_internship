#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) 
{
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
  
    mlir::PassRegistration<mlir::liferange::LifeRangePass>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Life Range Pass Driver", registry));
}