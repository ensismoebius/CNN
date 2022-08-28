cmake_minimum_required(VERSION 3.0.0)
project(CnnFromScratch VERSION 0.1.0)

include(CTest)
enable_testing()

# Submodule dependencies
add_subdirectory(src/lib/eigen)

add_executable(CnnFromScratch 
    src/main.cpp
    src/lib/utils/Rng.cpp
    src/lib/layer/Layer.cpp
    src/lib/output/Ouput.cpp
    src/lib/NeuralNetwork.cpp
    src/lib/optimizer/Optimizer.cpp
)

# Linking project to submodule dependencies
target_link_libraries(CnnFromScratch eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
