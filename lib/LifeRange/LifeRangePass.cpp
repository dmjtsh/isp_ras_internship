#include <stdio.h>
#include "../../include/LifeRange/Passes.h"
#include "../../../mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_LIFERANGE
	#include "../../include/LifeRange/Passes.h.inc"
} 
} 

using namespace mlir;
using namespace mlir::liferange;

namespace {

struct LifeRangePass
    : public affine::impl::LifeRangeBase<LifeRangePass> {

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
		printf("Hello Pass World");
	});
  }
};

} // namespace