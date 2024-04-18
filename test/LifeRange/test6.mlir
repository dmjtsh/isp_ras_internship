// RUN: lr-mlir-opt %s --life-range | FileCheck %s
/// hard (AliasAnalysis)
func.func @test6() {
  %alloc = memref.alloc() : memref<16xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+01 : f16
  %0 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %alloc) -> (memref<16xf16>) {
    memref.store %cst, %arg1[%arg0] : memref<16xf16>
    scf.yield %arg1 : memref<16xf16>
  }
  %alloc_0 = memref.alloc() : memref<10xi32>
  %1 = memref.load %0[%c1] : memref<16xf16>
  %2 = arith.addf %1, %cst : f16
  memref.store %2, %0[%c1] : memref<16xf16>
  memref.store %c1_i32, %alloc_0[%c1] : memref<10xi32>
  return
}

// CHECK: (0) memref_0: [0; 12]
// CHECK: (1) memref_2: [9; 13]

// CHECK: No memory to unite :-(
