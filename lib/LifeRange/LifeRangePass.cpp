#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Analysis/Liveness.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include "LifeRange/Passes.h"

namespace mlir {
namespace liferange {
#define GEN_PASS_DEF_LIFERANGE
	#include "LifeRange/Passes.h.inc"
} 
} 

using namespace mlir;
using namespace mlir::liferange;

 size_t SearchMaxLiveInd(Liveness::OperationListT live_operations, DenseMap<Operation *, size_t> operation_ids)
 {
   size_t max_ind = 0;
   for(size_t i = 0; i < live_operations.size(); i++)
     max_ind = std::max(max_ind, operation_ids[live_operations[i]]);

   return max_ind;
 }

 size_t SearchMinLiveInd(Liveness::OperationListT live_operations, DenseMap<Operation *, size_t> operation_ids)
 {
   size_t min_ind = operation_ids.size();
   for(size_t i = 0; i < live_operations.size(); i++)
     min_ind = std::min(min_ind, operation_ids[live_operations[i]]);

   return min_ind;
 }

//Prints Operand's Life Interval in brackets
void PrintOpLifeRange(Liveness* liveness) {
  llvm::raw_ostream& os = llvm::outs();

  DenseMap<Block *, size_t>     block_ids;
  DenseMap<Operation *, size_t> operation_ids;
  DenseMap<Value, size_t>       value_ids;
  
  liveness->getOperation()->walk<WalkOrder::PreOrder>([&](Block *block) {
    block_ids.insert({block, block_ids.size()});
    for (BlockArgument argument : block->getArguments())
      value_ids.insert({argument, value_ids.size()});
    for (Operation &operation : *block) {
      operation_ids.insert({&operation, operation_ids.size()});
      for (Value result : operation.getResults())
      {
        // Filling array with memref indexes for printing it
        if(isa<TypedValue<MemRefType>>(result))
          value_ids.insert({result, value_ids.size()});
      }
    }
  });

  // Lambda Function for Printing Memref's Name
  auto printValue = [&](Value value) {
    if (value.getDefiningOp())
      os << "memref_" << value_ids[value];
  };

  std::vector<std::pair<size_t, size_t>> values_intervals;
  liveness->getOperation()->walk<WalkOrder::PreOrder>([&](Block *block) {
    // Print liveness intervals.
    for (Operation &op : *block) {
      if (op.getNumResults() < 1)
        continue;
      for (Value result : op.getResults()) {
        // If Value is Memref
        if(isa<TypedValue<MemRefType>>(result))
        {
          printValue(result);
          os << ":";
          
          std::pair<size_t, size_t> value_interval;
          auto live_operations = liveness->resolveLiveness(result);
          value_interval.first  = SearchMinLiveInd(live_operations, operation_ids);
          value_interval.second = SearchMaxLiveInd(live_operations, operation_ids);
    
          os << "[" << value_interval.first << "; " << value_interval.second << "]\n";
        }
      }
    }
    os << "\n";
  });

}

namespace {

struct LifeRangePass
    : public liferange::impl::LifeRangeBase<LifeRangePass> {

  void runOnOperation() override {
    Liveness &lv = getAnalysis<Liveness>();
    
    llvm::raw_ostream& os = llvm::outs();

    lv.print(os);
    PrintOpLifeRange(&lv);
  }
  
};

}; // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::liferange::createLifeRangePass() {
  return std::make_unique<LifeRangePass>();
}