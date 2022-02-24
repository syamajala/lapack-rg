import "regent"
local c = regentlib.c

terralib.linklibrary("/home/vsyamaj/pkg/lib/liblapacke.so")
local lapacke = terralib.includec("lapacke.h", {"-I", "/home/vsyamaj/pkg/include/netlib"})

local lapack = terralib.includec("lapack_tasks.h", {"-I", "./build"})
terralib.linklibrary("./build/lapack_tasks.so")

terralib.linklibrary("./build/libcontext_manager.so")

require("lapack")


task main()
  --- Make sure to use single threaded blas
  -- cblas.openblas_set_num_threads(1)

  var M = 3
  var N = 2

  var A = region(ispace(int2d, {M, N}), double)
  var U = region(ispace(int2d, {M, M}), double)
  var VT = region(ispace(int2d, {N, N}), double)
  var S = region(ispace(int1d, {N}), double)

  fill(A, 0)
  fill(U, 0)
  fill(VT, 0)
  fill(S, 0)

  A[{0, 0}] = 1
  A[{0, 1}] = 2
  A[{1, 0}] = 4
  A[{1, 1}] = 5
  A[{2, 0}] = 2
  A[{2, 1}] = 1

  dgesvd(lapacke.LAPACK_COL_MAJOR, 'A', 'A', A, S, U, VT)

  c.printf("DGESVD\n")
  for i in A.ispace do
    c.printf("A[%d, %d] = %0.3f\n", i.x, i.y, A[i])
  end
  c.printf("\n")

  M = 3
  var B = region(ispace(int2d, {M, M}), double)

  B[{0, 0}] = 1
  B[{0, 1}] = 2
  B[{0, 2}] = 3

  B[{1, 0}] = 4
  B[{1, 1}] = 5
  B[{1, 2}] = 6

  B[{2, 0}] = 7
  B[{2, 1}] = 8
  B[{2, 2}] = 10

  var IPIV = region(ispace(int1d, M), int)
  fill(IPIV, 0)

  dgetrf(lapacke.LAPACK_COL_MAJOR, B, IPIV)

  c.printf("DGETRF\n")
  for i in B.ispace do
    c.printf("B[%d, %d] = %0.3f\n", i.x, i.y, B[i])
  end
  c.printf("\n")

  dgetri(lapacke.LAPACK_COL_MAJOR, B, IPIV)

  c.printf("DGETRI\n")
  for i in B.ispace do
    c.printf("B[%d, %d] = %0.3f\n", i.x, i.y, B[i])
  end

end

regentlib.start(main, lapack.lapack_tasks_h_register)
