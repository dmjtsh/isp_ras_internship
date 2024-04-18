// RUN: lr-mlir-opt %s --life-range | FileCheck %s
// normal
func.func @test3(%input : memref<16xf16>) {
  %alloc = memref.alloc() : memref<10xi32>

  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+01 : f16

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %cst) -> (f16) {
    %load = memref.load %input[%arg0] : memref<16xf16>
    memref.store %c1_i32, %alloc[%arg0] : memref<10xi32>

    %alloc_0 = memref.alloc() : memref<10xf16>
  
    %add = arith.addf %load, %arg1 : f16
    memref.store %add, %alloc_0[%c1] : memref<10xf16>

    scf.yield %add : f16
  }
  memref.store %0, %input[%c1] : memref<16xf16>
  return
}

// CHECK: (0) memref_0: [0; 12]
// CHECK-NEXT: (1) memref_1: [9; 11]
// CHECK-NEXT: (2) memref_arg0@0: [0; 14]
// CHECK-EMPTY:
// CHECK-NEXT: No memory to unite :-(
