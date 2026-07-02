cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 61)
project(cmake_and_cuda LANGUAGES CXX CUDA)
INCLUDE_DIRECTORIES("/home/marcugas/Documents/fork/GPU-synchrotron-compression/CUDA_ImCod/lib" "/usr/local/cuda-12.6/targets/x86_64-linux/include" "/usr/local/cuda-13.1/targets/x86_64-linux/include/nvtx3")
LINK_DIRECTORIES("/home/marcugas/Documents/fork/GPU-synchrotron-compression/CUDA_ImCod/lib" "/usr/local/cuda-12.6/targets/x86_64-linux/include" )

message("Printing Defaults")
message("Done Default")

set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")
message(${CMAKE_CUDA_HOST_COMPILER})

set (CMAKE_CUDA_FLAGS -rdc=true\ -maxrregcount=0\ --machine\ 64\ -cudart\ static\ -use_fast_math\ -Xptxas\ -v\ -lineinfo )
message(${CMAKE_CUDA_FLAGS})
set(CMAKE_CXX_FLAGS -fpermissive)
message(${CMAKE_CXX_FLAGS})

add_library(PICSONG_LIB STATIC
        Image/Image.cpp
        Image/Image.hpp
        IO/CommandLineParser.hpp
        IO/CommandLineParser.cpp
        IO/IOManager.hpp
        IO/IOManager.ipp
        SupportFunctions/AuxiliarFunctions.hpp
        SupportFunctions/AuxiliarFunctions.cpp

)
target_compile_features(PICSONG_LIB PUBLIC cxx_std_14)

add_executable(PICSONG Launcher.cu
  BitStreamBuilder/BitStreamBuilder.cpp
  BitStreamBuilder/BitStreamBuilder.cu
  BitStreamBuilder/BitStreamBuilder.cuh
  BitStreamBuilder/BitStreamBuilder.hpp
  BPC/BPCEngine.cuh
  BPC/BPCEngine.hpp
  BPC/BPCEngine.ipp
  Engines/CodingEngine.cu
  Engines/CodingEngine.cuh
  Engines/DecodingEngine.cu
  Engines/DecodingEngine.cuh
  Engines/Engine.cu
  Engines/Engine.cuh
  )

  set_target_properties(PICSONG PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
  set_target_properties(PICSONG PROPERTIES CUDA_STANDARD 14)

  target_link_libraries(PICSONG PUBLIC PICSONG_LIB nvToolsExt)

  install(TARGETS PICSONG DESTINATION bin)