cmake_minimum_required(VERSION 3.10)
project(Tensor_CFPQ CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -Xptxas -O3 -v")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 11)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61")

set(CFPQ_HEADERS
  cuda/kernels.h
  cuda/context.h
  cuda/misc.h
  cuda/tests_dev.h
  cuda/tests.h
  cuda/solution.h
  cuda/memory.h
  cuda/automata_loader.h
  cuda/graph_data.h
  cuda/timer.h
)

set(CFPQ_SOURCES
  main.cu
  ${CFPQ_HEADERS}
)

include_directories("${PROJECT_SOURCE_DIR}/cuda")

add_executable(
  tensor_cfpq
  ${CFPQ_HEADERS}
)