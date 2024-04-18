// RUN: lr-mlir-opt %s --life-range | FileCheck %s
// easy
func.func @test2() {
  %alloc = memref.alloc() : memref<16xf16>
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+01 : f16
  %1 = memref.load %alloc[%c1] : memref<16xf16>
  %2 = arith.addf %1, %cst : f16
  memref.store %2, %alloc[%c1] : memref<16xf16>
  %alloc_0 = memref.alloc() : memref<10xi32>
  memref.store %c1_i32, %alloc_0[%c1] : memref<10xi32>
  return
}
// CHECK: (0) memref_0: [0; 6]
// CHECK: (1) memref_1: [7; 8]

// CHECK: We can unite (0) and (1) memory!
