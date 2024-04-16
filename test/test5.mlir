func.func @test5(%input : memref<16xf16>) {
  %alloc = memref.alloc() : memref<10xi32>

  %c10 = arith.constant 10 : index
  %load1 = memref.load %alloc[%c10] : memref<10xi32>

  %c1_i32 = arith.constant 1 : i32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %pi = arith.constant 3.14 : f16
  %cst2 = arith.constant 35 : i32
  %0 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %pi) -> (f16) {
    %load2 = memref.load %input[%arg0] : memref<16xf16>

    %alloc_0 = memref.alloc() : memref<10xf16>
  
    %add = arith.addf %load2, %arg1 : f16
    memref.store %add, %alloc_0[%c1] : memref<10xf16>

    %alloc_1 = memref.alloc() : memref<10xi32>
    memref.store %cst2, %alloc_1[%c0] : memref<10xi32>

    scf.yield %add : f16
  }
  memref.store %0, %input[%c1] : memref<16xf16>

  %2 = arith.addi %cst2, %c1_i32 : i32
  memref.store %2, %alloc[%c1] : memref<10xi32>

  %alloc_ext = memref.alloc() : memref<15xf16>
  
  return
}