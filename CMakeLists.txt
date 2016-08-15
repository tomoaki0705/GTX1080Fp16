cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message("-- CUDA --")
    message("Version: " ${CUDA_VERSION})
    message("Library: " ${CUDA_CUDA_LIBRARY})
    message("Runtime: " ${CUDA_CUDART_LIBRARY})
    set(CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")
    set(CUDA_SOURCES multiply.cu)
else(CUDA_FOUND)
    message("Could not find CUDA in this system!")
endif(CUDA_FOUND)

add_executable(fp16Mul multiply.cpp ${CUDA_SOURCES})
