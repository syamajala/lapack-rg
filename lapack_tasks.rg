-- Copyright 2022 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

import "regent"
local clib = regentlib.c
local nan = regentlib.nan(double)
local utils = require("utils")

local lapack_h = utils.lapacke
require("lapacke")
if utils.use_gpu then
  require("cusolver")
end

__demand(__cuda, __leaf)
task dgesvd(
    layout : int,
    jobu   : &int8,
    jobvt  : &int8,
    A      : region(ispace(int2d), double),
    S      : region(ispace(int1d), double),
    U      : region(ispace(int2d), double),
    VT     : region(ispace(int2d), double))
where
  reads writes(A),
  writes(S, U, VT)
do
  var rectA = A.bounds
  var sizeA = rectA.hi - rectA.lo + {1, 1}
  var M = sizeA.x
  var N = sizeA.y
  var rectS = S.bounds
  var rectU = U.bounds
  var rectVT = VT.bounds

  var proc = get_executing_processor(__runtime())

  if clib.legion_processor_kind(proc) == clib.TOC_PROC then
    [(function()
          if utils.use_gpu then
            return rquote
                regentlib.assert(layout == lapack_h.LAPACK_COL_MAJOR, 'Expected column major layout.')
              return dgesvd_gpu_terra(layout, jobu, jobvt, M, N,
                                      rectA, __physical(A)[0], __fields(A)[0],
                                      rectS, __physical(S)[0], __fields(S)[0],
                                      rectU, __physical(U)[0], __fields(U)[0],
                                      rectVT, __physical(VT)[0], __fields(VT)[0])

                   end
          else
            return rquote regentlib.assert(false, "Build with CUDA support.") end
          end
     end)()]
  else
    return dgesvd_cpu_terra(layout, jobu, jobvt, M, N,
                            rectA, __physical(A)[0], __fields(A)[0],
                            rectS, __physical(S)[0], __fields(S)[0],
                            rectU, __physical(U)[0], __fields(U)[0],
                            rectVT, __physical(VT)[0], __fields(VT)[0])
  end
end

__demand(__cuda, __leaf)
task dgetrf(
    layout : int,
    A      : region(ispace(int2d), double),
    IPIV   : region(ispace(int1d), int))
where
  reads writes(A),
  writes(IPIV)
do
  var rectA = A.bounds
  var sizeA = rectA.hi - rectA.lo + {1, 1}
  var M = sizeA.x
  var N = sizeA.y

  var rectIPIV = IPIV.bounds

  var proc = get_executing_processor(__runtime())

  if clib.legion_processor_kind(proc) == clib.TOC_PROC then
    [(function()
          if utils.use_gpu then
            return rquote
                regentlib.assert(layout == lapack_h.LAPACK_COL_MAJOR, 'Expected column major layout.')
              return dgetrf_gpu_terra(layout, M, N,
                                      rectA, __physical(A)[0], __fields(A)[0],
                                      rectIPIV, __physical(IPIV)[0], __fields(IPIV)[0])
                   end
          else
            return rquote regentlib.assert(false, "Build with CUDA support.") end
          end
     end)()]
  else
    return dgetrf_cpu_terra(layout, M, N,
                            rectA, __physical(A)[0], __fields(A)[0],
                            rectIPIV, __physical(IPIV)[0], __fields(IPIV)[0])
  end
end

__demand(__cuda, __leaf)
task dgetri(
    layout : int,
    A      : region(ispace(int2d), double),
    IPIV   : region(ispace(int1d), int))
where
  reads writes(A),
  reads(IPIV)
do
  var rectA = A.bounds
  var sizeA = rectA.hi - rectA.lo + {1, 1}
  var N = sizeA.x

  var rectIPIV = IPIV.bounds

  var proc = get_executing_processor(__runtime())

  if clib.legion_processor_kind(proc) == clib.TOC_PROC then
    [(function()
          if utils.use_gpu then
            return rquote
                regentlib.assert(layout == lapack_h.LAPACK_COL_MAJOR, 'Expected column major layout.')
              return dgetri_gpu_terra(layout, N,
                                      rectA, __physical(A)[0], __fields(A)[0],
                                      rectIPIV, __physical(IPIV)[0], __fields(IPIV)[0])
                   end
          else
            return rquote regentlib.assert(false, "Build with CUDA support.") end
          end
     end)()]
  else
    return dgetri_cpu_terra(layout, N,
                            rectA, __physical(A)[0], __fields(A)[0],
                            rectIPIV, __physical(IPIV)[0], __fields(IPIV)[0])
  end
end

local tasks_h = utils.output_dir .. "/lapack_tasks.h"
local tasks_so = utils.output_dir .. "/lapack_tasks.so"
regentlib.save_tasks(tasks_h, tasks_so, nil, nil, nil, nil, false)
