func.func @allocate_buffers() {
  %buf1 = memref.alloc() : memref<100x100xf32>
  %buf2 = memref.alloc() : memref<200x200xf32>
  
  memref.dealloc %buf1 : memref<100x100xf32>
  memref.dealloc %buf2 : memref<200x200xf32>

  return
}