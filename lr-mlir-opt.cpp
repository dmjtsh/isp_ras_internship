#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) 
{
  mlir::registerAllDialects();
  
  llvm::StringRef toolName = "lr-mlir-opt";

  mlir::DialectRegistry registry;

  return mlir::MlirOptMain(argc, argv, toolName, registry)
                 .succeeded()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}