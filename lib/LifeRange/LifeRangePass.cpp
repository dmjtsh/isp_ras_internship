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
#include <string>
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

void LifeAliasAnalysis(
    AliasAnalysis *alias, std::vector<Value> *memrefs_to_print,
    std::vector<std::pair<size_t, size_t>> *values_intervals) {

  for (int i = 0; i < (int)memrefs_to_print->size(); i++) {
    for (int j = i + 1; j < (int)memrefs_to_print->size(); j++) {
      if (alias->alias((*memrefs_to_print)[i], (*memrefs_to_print)[j]) ==
          AliasResult::MustAlias) {
        if ((*values_intervals)[i].first > (*values_intervals)[j].first)
          (*values_intervals)[i].first = (*values_intervals)[j].first;

        if ((*values_intervals)[i].second < (*values_intervals)[j].second)
          (*values_intervals)[i].second = (*values_intervals)[j].second;

        // Resting Old Pointer
        memrefs_to_print->erase(memrefs_to_print->begin() + j);
        values_intervals->erase(values_intervals->begin() + j);

        i--;
        break;
      }
    }
  }
}

// Prints  Values Life Intervals in brackets
// Returns Values Life Ranges
std::vector<std::pair<size_t, size_t>>
PrintValuesLifeRanges(Liveness *liveness, AliasAnalysis *alias) {
  DenseMap<Block *, size_t> block_ids;
  DenseMap<Operation *, size_t> operation_ids;
  DenseMap<Value, size_t> value_ids;

  liveness->getOperation()->walk<WalkOrder::PreOrder>([&](Block *block) {
    block_ids.insert({block, block_ids.size()});
  });

  liveness->getOperation()->walk<WalkOrder::PreOrder>(
      [&](Operation *operation) {
        operation_ids.insert({operation, operation_ids.size() - 1});
        for (Value result : operation->getResults()) {
          // Filling array with memref indexes for printing it
          if (isa<TypedValue<MemRefType>>(result))
            value_ids.insert({result, value_ids.size()});
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

  std::vector<Value> memrefs_to_print(value_ids.size());
  std::vector<std::pair<size_t, size_t>> values_intervals(value_ids.size());
  std::pair<size_t, size_t> result_interval;

  liveness->getOperation()->walk<WalkOrder::PreOrder>([&](Block *block) {
    // Adding Memref Arguments of blocks
    for (Value arg : block->getArguments()) {
      if (isa<TypedValue<MemRefType>>(arg)) {
        result_interval.first = operation_ids[&block->front()];
        result_interval.second = operation_ids[&block->back()];

        memrefs_to_print.push_back(arg);
        values_intervals.push_back(result_interval);
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
                values_intervals[value_ids[value]].second;

            // Setting New Maximum Interval for a var that is in Cycle Block
            if (current_value_max_ind < cycle_block_end_ind)
              values_intervals[value_ids[value]].second = cycle_block_end_ind;
          }
        }
      }

      for (Value result : op.getResults()) {
        // If Value is Memref
        if (isa<TypedValue<MemRefType>>(result)) {
          auto live_operations = liveness->resolveLiveness(result);
          result_interval.first =
              SearchMinLiveInd(live_operations, operation_ids);
          result_interval.second =
              SearchMaxLiveInd(live_operations, operation_ids);

          // Setting Value And Interval of Value with Its Index
          size_t result_index = value_ids[result];
          values_intervals[result_index] = result_interval;
          memrefs_to_print[result_index] = result;
        }
      }
    }
  });

  LifeAliasAnalysis(alias, &memrefs_to_print, &values_intervals);

  // Printing all Other Memrefs
  for (size_t i = 0; i < memrefs_to_print.size(); i++)
    printMemref(memrefs_to_print[i], values_intervals[i]);

  return values_intervals;
}

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
    AliasAnalysis &aa = getAnalysis<AliasAnalysis>();

    llvm::raw_ostream &os = llvm::outs();
    lv.print(os);

    llvm::outs() << "\n----------LifeRangePass----------\n\n";
    std::vector<std::pair<size_t, size_t>> life_ranges =
        PrintValuesLifeRanges(&lv, &aa);
    PrintIndependentLifeRanges(life_ranges);
    llvm::outs() << "----------LifeRangePass----------\n\n";
  }
};

}; // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::liferange::createLifeRangePass() {
  return std::make_unique<LifeRangePass>();
}