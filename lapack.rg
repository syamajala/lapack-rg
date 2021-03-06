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

extern
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
end

extern
task dgetrf(
    layout : int,
    A      : region(ispace(int2d), double),
    IPIV   : region(ispace(int1d), int))
where
  reads writes(A),
  writes(IPIV)
end

extern
task dgetri(
    layout : int,
    A      : region(ispace(int2d), double),
    IPIV   : region(ispace(int1d), int))
where
  reads writes(A),
  reads(IPIV)
end
