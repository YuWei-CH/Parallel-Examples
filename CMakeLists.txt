cmake_minimum_required(VERSION 3.16)
project(vecadd C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3 -march=native -mtune=native -fstrict-aliasing -fopt-info-loop-optimized")
set(CMAKE_NOVEC_FLAGS "-fno-tree-vectorize")

find_package(OpenMP REQUIRED)

# List of source files (excluding header)
set(SRC timer.c)

# Helper macro to define executables
function(add_vec_target name source)
  add_executable(${name} ${source} ${SRC})
  target_compile_options(${name} PRIVATE ${CMAKE_NOVEC_FLAGS} ${OpenMP_C_FLAGS})
  target_link_libraries(${name} PRIVATE OpenMP::OpenMP_C)
endfunction()

# Define all targets
add_vec_target(vecadd_opt  vecadd_opt.c)
add_vec_target(vecadd_opt1  vecadd_opt1.c)
add_vec_target(vecadd_opt2  vecadd_opt2.c)
# add_vec_target(vecadd_opt3  vecadd_opt3.c)

# Clean build cache
add_custom_target(distclean
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake_clean.cmake
  COMMAND rm -rf CMakeCache.txt CMakeFiles Makefile cmake_install.cmake
)