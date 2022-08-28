cmake_minimum_required(VERSION 3.0.0)
project(CnnFromScratch VERSION 0.1.0)

include(CTest)
enable_testing()

# Submodule dependencies
add_subdirectory(src/lib/eigen)

add_executable(CnnFromScratch src/main.cpp)

# Linking project to submodule dependencies
target_link_libraries(CnnFromScratch eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
