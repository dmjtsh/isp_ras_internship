#include <stdio.h>

#include "LifeRange/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace liferange {
#define GEN_PASS_DEF_LIFERANGE
	#include "LifeRange/Passes.h.inc"
} 
} 

using namespace mlir;
using namespace mlir::liferange;

namespace {

struct LifeRangePass
    : public liferange::impl::LifeRangeBase<LifeRangePass> {

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
		printf("Hello Pass World");
	});
  }
};

} // namespace