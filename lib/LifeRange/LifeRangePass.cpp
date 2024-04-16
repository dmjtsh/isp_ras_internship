#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/Liveness.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include "LifeRange/Passes.h"
#include <algorithm>
#include <iostream>

namespace mlir {
namespace liferange {
#define GEN_PASS_DEF_LIFERANGE

#include "LifeRange/Passes.h.inc"

} // namespace liferange
} // namespace mlir

using namespace mlir;
using namespace mlir::liferange;

size_t SearchMaxLiveInd(Liveness::OperationListT live_operations,
                        DenseMap<Operation *, size_t> operation_ids) {
  size_t max_ind = 0;
  for (size_t i = 0; i < live_operations.size(); i++)
    max_ind = std::max(max_ind, operation_ids[live_operations[i]]);

  return max_ind;
}

size_t SearchMinLiveInd(Liveness::OperationListT live_operations,
                        DenseMap<Operation *, size_t> operation_ids) {
  size_t min_ind = operation_ids.size();
  for (size_t i = 0; i < live_operations.size(); i++)
    min_ind = std::min(min_ind, operation_ids[live_operations[i]]);

  return min_ind;
}

// Prints  Values Life Intervals in brackets
// Returns Values Life Ranges
std::vector<std::pair<size_t, size_t>>
PrintValuesLifeRanges(Liveness *liveness) {
  DenseMap<Block *, size_t> block_ids;
  DenseMap<Operation *, size_t> operation_ids;
  DenseMap<Value, size_t> value_ids;

  liveness->getOperation()->walk<WalkOrder::PostOrder>([&](Block *block) {
    block_ids.insert({block, block_ids.size()});
    for (Operation &operation : *block) {
      
      // DEBUG
      std::cout << "  <--- Operation #" << operation_ids.size(); 
      llvm::raw_ostream &os = llvm::outs();
      operation.print(os);
      std::cout << "\n";
      // DEBUG

      operation_ids.insert({&operation, operation_ids.size()});
      for (Value result : operation.getResults()) {
        // Filling array with memref indexes for printing it
        if (isa<TypedValue<MemRefType>>(result))
          value_ids.insert({result, value_ids.size()});
      }
    }
  });

  // Lambda Function for Printing Memref's Name
  auto printMemref = [&](Value value, std::pair<size_t, size_t> interval) {
    if (value.getDefiningOp())
      llvm::outs() << "memref_" << value_ids[value];
    else {
      auto block_arg = cast<BlockArgument>(value);
      llvm::outs() << "memref_arg" << block_arg.getArgNumber() << "@"
                   << block_ids[block_arg.getOwner()];
    }

    llvm::outs() << ": [" << interval.first << "; " << interval.second << "]\n";
  };

  std::vector<std::pair<size_t, size_t>> values_intervals(value_ids.size());
  std::pair<size_t, size_t> result_interval;

  liveness->getOperation()->walk<WalkOrder::PreOrder>([&](Block *block) {
    // Print Memref Arguments of blocks
    for (Value arg : block->getArguments()) {
      if (isa<TypedValue<MemRefType>>(arg)) {
        result_interval.first = operation_ids[&block->front()];
        result_interval.second = operation_ids[&block->back()];

        printMemref(arg, result_interval);
      }
    }

    // Print liveness intervals.
    for (Operation &op : *block) {
      if (op.getNumResults() < 1)
        continue;
      for (Value result : op.getResults()) {
        // If Value is Memref
        if (isa<TypedValue<MemRefType>>(result)) {
          auto live_operations = liveness->resolveLiveness(result);
          result_interval.first =
              SearchMinLiveInd(live_operations, operation_ids);
          result_interval.second =
              SearchMaxLiveInd(live_operations, operation_ids);

          printMemref(result, result_interval);

          // Setting Interval of Value with Its Index
          size_t result_index = value_ids[result];
          values_intervals[result_index] = result_interval;
        }
      }
    }
  });

  return values_intervals;
}

// Prints Independent Life Ranges of Values
void PrintIndependentLifeRanges(
    std::vector<std::pair<size_t, size_t>> life_ranges) {
  bool memory_can_be_united = false;
  for (size_t i = 0; i < life_ranges.size(); i++) {
    for (size_t j = i + 1; j < life_ranges.size(); j++) {
      // No Intersection between to intervals
      if (life_ranges[i].second < life_ranges[j].first ||
          life_ranges[j].second < life_ranges[i].first) {
        memory_can_be_united = true;
        llvm::outs() << "We can unite \"memref_" << i << "\" and "
                     << "\"memref_" << j << "\" memory!\n";
      }
    }
  }

  if (!memory_can_be_united)
    llvm::outs() << "No memory to unite :-(\n";

  llvm::outs() << "\n";
}

namespace {

struct LifeRangePass : public liferange::impl::LifeRangeBase<LifeRangePass> {

  void runOnOperation() override {
    Liveness &lv = getAnalysis<Liveness>();

    llvm::raw_ostream &os = llvm::outs();
    lv.print(os);

    llvm::outs() << "\n----------LifeRangePass----------\n\n";
    std::vector<std::pair<size_t, size_t>> life_ranges =
        PrintValuesLifeRanges(&lv);
    PrintIndependentLifeRanges(life_ranges);
    llvm::outs() <<   "----------LifeRangePass----------\n\n";
  }
};

}; // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::liferange::createLifeRangePass() {
  return std::make_unique<LifeRangePass>();
}