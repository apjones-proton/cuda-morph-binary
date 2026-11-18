cmake_minimum_required(VERSION 3.24)

project(cuda_morphology
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

#set(CMAKE_CUDA_ARCHITECTURES 75)
#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#    set(CMAKE_CUDA_ARCHITECTURES native)
#endif()

add_library(cuda_morphology
    src/morphology.cu
)

add_library(cuda_morphology::cuda_morphology
    ALIAS cuda_morphology
)

target_include_directories(cuda_morphology
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(cuda_morphology PUBLIC cxx_std_17)

set_target_properties(cuda_morphology PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuda_morphology
    PUBLIC CUDA::cudart
)

option(CUDA_MORPHOLOGY_BUILD_TESTS
    "Build tests" OFF)


if(CUDA_MORPHOLOGY_BUILD_TESTS)
    add_subdirectory(testing)
endif()
