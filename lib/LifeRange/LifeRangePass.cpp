#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Analysis/Liveness.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>

#include "LifeRange/Passes.h"

#include <algorithm>

namespace mlir {
namespace liferange {
#define GEN_PASS_DEF_LIFERANGE

#include "LifeRange/Passes.h.inc"

} // namespace liferange
} // namespace mlir

using namespace mlir;
using namespace mlir::liferange;

/*
* Life Interval Struct [start; end]
*/
struct LifeInterval
{
  size_t start;
  size_t end;
};

/*
 * Life Alias Analysis function
 * Checks all memrefs if they are Alias or not
 * Then changes memref vector and values intervals uniting all alias intervals
 */
void LifeAliasAnalysis(
    AliasAnalysis *alias, std::vector<Value> *memrefs_to_print,
    std::vector<LifeInterval> *values_intervals) {

  for (int i = 0; i < (int)memrefs_to_print->size(); i++) {
    for (int j = i + 1; j < (int)memrefs_to_print->size(); j++) {
      if (alias->alias((*memrefs_to_print)[i], (*memrefs_to_print)[j]) ==
          AliasResult::MustAlias) {
        if ((*values_intervals)[i].start > (*values_intervals)[j].start)
          (*values_intervals)[i].start = (*values_intervals)[j].start;

        if ((*values_intervals)[i].end < (*values_intervals)[j].end)
          (*values_intervals)[i].end = (*values_intervals)[j].end;

        // Leaving Oldest Memref
        memrefs_to_print->erase(memrefs_to_print->begin() + j);
        values_intervals->erase(values_intervals->begin() + j);

        // Just to check all of the Aliases
        i--;
        break;
      }
    }
  }
}

/*
* Get Max/Min Live Index functions
* Function for getting an index of First Operation in Life Range
* to build proper Life Range Intervals of values
*/
size_t GetMaxLiveInd(Liveness::OperationListT live_operations,
                     DenseMap<Operation *, size_t> operation_ids) {
  size_t max_ind = 0;
  for (size_t i = 0; i < live_operations.size(); i++)
    max_ind = std::max(max_ind, operation_ids[live_operations[i]]);

  return max_ind;
}
size_t GetMinLiveInd(Liveness::OperationListT live_operations,
                     DenseMap<Operation *, size_t> operation_ids) {
  size_t min_ind = operation_ids.size();
  for (size_t i = 0; i < live_operations.size(); i++)
    min_ind = std::min(min_ind, operation_ids[live_operations[i]]);

  return min_ind;
}

/*
* Print Values Life Ranges function
* Scans all memrefs and Prints Life Intervals in Brackets
* Returns Vector of Life Intervals 
*/
std::vector<LifeInterval>
PrintValuesLifeRanges(Liveness *liveness, AliasAnalysis *alias) {
  // Arrays of Indexes
  DenseMap<Block *, size_t>     block_ids;
  DenseMap<Operation *, size_t> operation_ids;
  DenseMap<Value, size_t>       value_ids;

  liveness->getOperation()->walk<WalkOrder::PreOrder>(
      [&](Operation *operation) {
        operation_ids.insert({operation, operation_ids.size() - 1});
        for (Value result : operation->getResults()) {
          if (isa<TypedValue<MemRefType>>(result))
            value_ids.insert({result, value_ids.size()});
        }
      });
  liveness->getOperation()->walk<WalkOrder::PreOrder>([&](Block *block) {
    for (Value arg : block->getArguments()) {
      if (isa<TypedValue<MemRefType>>(arg))
        value_ids.insert({arg, value_ids.size()});
    }
    block_ids.insert({block, block_ids.size()});
  });

  std::vector<Value> memrefs_to_print(value_ids.size());
  // Array of Values, Indexing in this Array is Done By value_ids!
  std::vector<LifeInterval> values_intervals(value_ids.size());
  LifeInterval result_interval;

  liveness->getOperation()->walk<WalkOrder::PreOrder>([&](Block *block) {
    // Adding Memref Arguments of blocks
    for (Value arg : block->getArguments()) {
      if (isa<TypedValue<MemRefType>>(arg)) {
        result_interval.start = operation_ids[&block->front()];
        result_interval.end = operation_ids[&block->back()];

        // Adding Args in the end of memref array for comfortable printing
        values_intervals[value_ids[arg]] = result_interval;
        memrefs_to_print[value_ids[arg]] = arg;
      }
    }

    for (Operation &op : *block) {
      if (op.getNumResults() < 1)
        continue;

      // Checking if we are in Cycle Block
      if (isa<scf::ForOp>(op)) {
        const LivenessBlockInfo *block_liveness = liveness->getLiveness(block);
        auto currently_live_values = block_liveness->currentlyLiveValues(&op);

        // Last Block of Cycle
        Block &cycle_block = op.getRegions().back().back();
        size_t cycle_block_end_ind = operation_ids[&cycle_block.back()];

        for (Value value : currently_live_values) {
          if (isa<TypedValue<MemRefType>>(value)) {
            size_t current_value_max_ind =
                values_intervals[value_ids[value]].end;

            // Setting New Maximum Interval for a var that is in Cycle Block
            if (current_value_max_ind < cycle_block_end_ind)
              values_intervals[value_ids[value]].end = cycle_block_end_ind;
          }
        }
      }

      for (Value result : op.getResults()) {
        if (isa<TypedValue<MemRefType>>(result)) {
          auto live_operations = liveness->resolveLiveness(result);
          result_interval.start  = GetMinLiveInd(live_operations, operation_ids);
          result_interval.end = GetMaxLiveInd(live_operations, operation_ids);

          // Setting Value And Interval of Value with Its Index
          values_intervals[value_ids[result]] = result_interval;
          memrefs_to_print[value_ids[result]] = result;
        }
      }
    }
  });

  LifeAliasAnalysis(alias, &memrefs_to_print, &values_intervals);

  // Lambda Function for Printing Memref's Name
  auto printMemref = [&](Value value, LifeInterval interval,
                         size_t memref_num) {
    llvm::outs() << "(" << memref_num << ") ";
    if (value.getDefiningOp())
      llvm::outs() << "memref_" << value_ids[value];
    else {
      auto block_arg = cast<BlockArgument>(value);
      llvm::outs() << "memref_arg" << block_arg.getArgNumber() << "@"
                   << block_ids[block_arg.getOwner()];
    }

    llvm::outs() << ": [" << interval.start << "; " << interval.end << "]\n";
  };

  // Printing all Memrefs
  for (size_t i = 0; i < memrefs_to_print.size(); i++)
    printMemref(memrefs_to_print[i], values_intervals[i], i);

  return values_intervals;
}

/*
* Print Independent Life Ranges Function
* Scans for Intersection between intervals
* Prints non-Intersected intervals (so we can see where we can unite our memory)
*/
void PrintIndependentLifeRanges(
    std::vector<LifeInterval> life_ranges) {
  bool memory_can_be_united = false;
  for (size_t i = 0; i < life_ranges.size(); i++) {
    for (size_t j = i + 1; j < life_ranges.size(); j++) {
      // No Intersection between to intervals
      if (life_ranges[i].end < life_ranges[j].start ||
          life_ranges[j].end < life_ranges[i].start) {
        memory_can_be_united = true;
        llvm::outs() << "We can unite (" << i << ") and "
                     << "(" << j << ") memory!\n";
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
    AliasAnalysis &aa = getAnalysis<AliasAnalysis>();

    llvm::outs() << "\n----------LifeRangePass----------\n\n";
    std::vector<LifeInterval> life_ranges =
        PrintValuesLifeRanges(&lv, &aa);
    
    llvm::outs() << "\n";

    PrintIndependentLifeRanges(life_ranges);
    llvm::outs() <<   "----------LifeRangePass----------\n\n";
  }
};

}; // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::liferange::createLifeRangePass() {
  return std::make_unique<LifeRangePass>();
}