// easy
func.func @test1() {
  %alloc = memref.alloc() : memref<16xf16>
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+01 : f16
  %alloc_0 = memref.alloc() : memref<10xi32>
  %1 = memref.load %alloc[%c1] : memref<16xf16>
  %2 = arith.addf %1, %cst : f16
  memref.store %2, %alloc[%c1] : memref<16xf16>
  memref.store %c1_i32, %alloc_0[%c1] : memref<10xi32>
  return
}