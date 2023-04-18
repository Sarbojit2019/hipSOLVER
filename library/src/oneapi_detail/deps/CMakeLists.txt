set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations")
set(POSITION_INDEPENDENT_CODE ON)

add_library(sycl_solver_wrapper SHARED ./common.cpp ./sycl_solver_wrapper.cpp)

target_link_libraries(sycl_solver_wrapper PUBLIC sycl MKL::MKL mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_sequential MKL::mkl_core)
target_include_directories(sycl_solver_wrapper
                            PUBLIC  $<BUILD_INTERFACE:${DPCPP_INCLUDE_DIR}/sycl>
                            PUBLIC  $<BUILD_INTERFACE:${DPCPP_INCLUDE_DIR}>
                            PUBLIC  $<BUILD_INTERFACE:~/packages>
                            PRIVATE ${PROJECT_BINARY_DIR}/include/hipblas
                            PRIVATE $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. "$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_options(sycl_solver_wrapper PRIVATE "$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>")

# Install sycl_wrapper lib
install(TARGETS sycl_solver_wrapper DESTINATION ${CMAKE_INSTALL_LIBDIR})