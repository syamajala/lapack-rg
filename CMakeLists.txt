cmake_minimum_required(VERSION 3.8)
project(lapack-rg LANGUAGES CXX)

set(Legion_DIR $ENV{LG_RT_DIR}/../language/build)

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

find_package(LAPACK REQUIRED)

find_path(LAPACK_INCLUDE_DIRS lapacke.h
  /usr/include
  /usr/local/include
  $ENV{LAPACK_HOME}/include)

set(RegentFlags --lapack-header ${LAPACK_INCLUDE_DIRS} --lapack-library ${LAPACK_LIBRARIES} --output-dir ${PROJECT_BINARY_DIR} -fseparate 1)

add_custom_target(lapack_tasks ALL
  DEPENDS lapack_tasks.so)
set(OUTPUTS lapack_tasks.so lapack_tasks.h lapacke_tasks.so lapacke_tasks.h)
set(SRCS lapack.rg lapack_tasks.rg lapacke.rg utils.rg)

option(USE_CUDA "Enable support for CUDA" OFF)
if(USE_CUDA)
  message("Building CUSOLVER")
  enable_language(CUDA)
  add_library(lapack_context_manager SHARED lapack_context_manager.cu)
  target_link_libraries(lapack_context_manager -lcusolver)
  LIST(APPEND RegentFlags --use-gpu)
  LIST(APPEND OUTPUTS cusolver_tasks.so cusolver_tasks.h)
  LIST(APPEND SRCS lapack_context_manager.cu lapack_context_manager.h cusolver.rg)
  add_dependencies(lapack_tasks lapack_context_manager)
endif()

add_custom_command(OUTPUT lapack_tasks.so
  COMMAND regent.py ${CMAKE_CURRENT_LIST_DIR}/lapack_tasks.rg ${RegentFlags}
  DEPENDS ${SRCS}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  BYPRODUCTS ${OUTPUTS}
  VERBATIM)

add_custom_command(TARGET lapack_tasks POST_BUILD COMMAND cp ${CMAKE_SOURCE_DIR}/lapack.rg ${PROJECT_BINARY_DIR})
