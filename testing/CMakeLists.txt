find_package(OpenCV REQUIRED)

add_executable(tests
        tests.cpp
        cpu_morphology.cpp cases.cpp
        )

target_link_libraries(tests
        PRIVATE
        cuda_morphology::cuda_morphology
        ${OpenCV_LIBS}
        )

target_include_directories(tests
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        )


add_executable(benchmark
        benchmark.cpp
        cpu_morphology.cpp cases.cpp)

target_link_libraries(benchmark
        PRIVATE
        cuda_morphology::cuda_morphology
        CUDA::nppim
        CUDA::nppc
        )
