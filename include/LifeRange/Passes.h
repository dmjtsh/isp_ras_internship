#ifndef LIFE_RANGE_PASSES_H
#define LIFE_RANGE_PASSES_H

#include "mlir/Pass/Pass.h"
#include <limits>

namespace mlir {

namespace func {
class FuncOp;
} // namespace func

namespace liferange {

#define GEN_PASS_DECL

#include "LifeRange/Passes.h.inc"

std::unique_ptr<OperationPass<func::FuncOp>> 
	createLifeRangePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION

#include "LifeRange/Passes.h.inc"

} // namespace liferange
} // namespace mlir

#endif // LIFE_RANGE_PASSES_H
