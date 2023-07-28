# ########################################################################
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

# Linking lapack library requires fortran flags
#find_package( cblas REQUIRED CONFIG)
if (cblas_FOUND)
  message("cBLAS found")
else()
  message("Did not findout cBLAS")
endif()

find_package( OpenBLAS REQUIRED CONFIG)
if (OpenBLAS_FOUND)
  message("BLAS found")
else()
  message("Not found")
endif()

if( NOT TARGET hipsolver )
  find_package( hipsolver REQUIRED CONFIG PATHS /opt/rocm/hipsolver )
endif( )

find_package( GTest REQUIRED )

set(hipsolver_test_source
  hipsolver_gtest_main.cpp
    getrs_gtest.cpp       # All test passed
    getrf_gtest.cpp       # All test passed
    gebrd_gtest.cpp       # All test passed
    geqrf_gtest.cpp       # All test passed
    potrf_gtest.cpp       # test passed for float but not supported for double on en12
    potri_gtest.cpp       # test passed for float but not supported for double on en12
    potrs_gtest.cpp       # All test passed
    syevd_heevd_gtest.cpp # All test passed
    sygvd_hegvd_gtest.cpp # All test passed
    sytrd_hetrd_gtest.cpp # All test passed
    orgqr_ungqr_gtest.cpp # All test passed
    ormqr_unmqr_gtest.cpp # All test passed
  #gels_gtest.cpp
  #gesv_gtest.cpp
  #gesvd_gtest.cpp    # Spec says only for CPU
  #gesvda_gtest.cpp
  #gesvdj_gtest.cpp
  #syevj_heevj_gtest.cpp #not implemented
  #sygvj_hegvj_gtest.cpp
  #syevdx_heevdx_gtest.cpp
  #sygvdx_hegvdx_gtest.cpp
  #sytrf_gtest.cpp
  #orgbr_ungbr_gtest.cpp #Not supported error is thrown on Gen12
  #orgtr_ungtr_gtest.cpp #Not supported error is thrown on Gen12
  #ormtr_unmtr_gtest.cpp
)

set( hipsolver_test_common
  ../common/lapack_host_reference.cpp
  ../common/hipsolver_datatype2string.cpp
  ../common/utility.cpp
)

add_executable( hipsolver-test ${hipsolver_test_source} ${hipsolver_test_common} )
if( UNIX AND NOT USE_ONEAPI)
  target_sources( hipsolver-test PRIVATE ${hipsolver_f90_source} )
endif( )

target_include_directories( hipsolver-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

add_armor_flags( hipsolver-test "${ARMOR_LEVEL}" )

target_compile_definitions( hipsolver-test PRIVATE GOOGLE_TEST )

# External header includes included as SYSTEM files
target_include_directories( hipsolver-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

#target_link_libraries( hipsolver-test PRIVATE cblas lapack blas GTest::GTest Threads::Threads roc::hipsolver )
target_link_libraries( hipsolver-test PRIVATE lapack blas GTest::GTest Threads::Threads roc::hipsolver )
#if( UNIX )
#  target_link_libraries( hipsolver-test PRIVATE hipsolver_fortran_client )
#endif( )

target_link_libraries(hipsolver-test PRIVATE
  $<BUILD_INTERFACE:hipsolver-common>
)

# need mf16c flag for float->half convertion
target_compile_options( hipsolver-test PRIVATE -mf16c )

if( NOT USE_CUDA )
  target_link_libraries( hipsolver-test PRIVATE hip::host )

  if( CUSTOM_TARGET )
    target_link_libraries( hipsolver-test PRIVATE hip::${CUSTOM_TARGET} )
  endif( )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries( hipsolver-test PRIVATE -lpthread -lm )
  endif()
else( )
  target_compile_definitions( hipsolver-test PRIVATE __HIP_PLATFORM_NVCC__ )

  target_include_directories( hipsolver-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( hipsolver-test PRIVATE ${CUDA_LIBRARIES} Threads::Threads )
endif( )

set_target_properties( hipsolver-test PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( hipsolver-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

#rocm_install(TARGETS hipsolver-test COMPONENT tests)

if(WIN32)
  file(GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${cblas_DIR}/bin/*.dll
    ${GTest_DIR}/bin/*.dll
    $ENV{rocblas_DIR}/bin/*.dll
    $ENV{rocsolver_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
  )
  foreach(file_i ${third_party_dlls})
    add_custom_command(TARGET hipsolver-test
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/
    )
  endforeach()
  add_custom_command(TARGET hipsolver-test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory $ENV{rocblas_DIR}/bin/rocblas/library ${PROJECT_BINARY_DIR}/staging/library
  )
endif()
