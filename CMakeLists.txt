cmake_minimum_required(VERSION 3.16)
project(ass02-ps LANGUAGES CXX CUDA VERSION 1.1)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CC "89" CACHE STRING "Which compute capability to build for? e.g. 75, 86, 89, ...")


####################################################################
# Helper Functions
add_library(prefixsum_common
	src/dataset.cpp
	include/dataset.h
)
target_include_directories(prefixsum_common PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(prefixsum_common PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(prefixsum_common PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(prefixsum_common PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(prefixsum_common PRIVATE -Wall -Wextra $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

####################################################################
# CPU Implementation
add_library(prefixsum_cpu
	src/prefixsum.cpp
	include/prefixsum.h
)
target_include_directories(prefixsum_cpu PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(prefixsum_cpu PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(prefixsum_cpu PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(prefixsum_cpu PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(prefixsum_cpu PRIVATE -Wall -Wextra $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

####################################################################
# GPU Implementation
add_library(prefixsum_gpu
	include/prefixsum.h
	src/prefixsum.cu
)
target_include_directories(prefixsum_gpu PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(prefixsum_gpu PROPERTIES CUDA_ARCHITECTURES ${CC})
set_target_properties(prefixsum_gpu PROPERTIES CXX_STANDARD 17)
target_compile_options(prefixsum_gpu PRIVATE -Xcompiler -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -keep --ptxas-options=-v -lineinfo $<$<CONFIG:DEBUG>:-G>)


####################################################################
# Executable
add_executable(prefixsum
	src/main.cpp
	include/prefixsum.h
)
target_include_directories(prefixsum PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/helper")
set_target_properties(prefixsum PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(prefixsum PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(prefixsum PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(prefixsum PRIVATE -Wall -Wextra $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

# Link all together
target_link_libraries(prefixsum prefixsum_common prefixsum_cpu prefixsum_gpu)
