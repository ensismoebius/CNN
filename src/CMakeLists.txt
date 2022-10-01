cmake_minimum_required(VERSION 3.0.0)
project(TesteNeuro VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Armadillo REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(TesteNeuro
src/main.cpp
src/lib/NN.cpp
)

target_link_libraries(TesteNeuro ${ARMADILLO_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
