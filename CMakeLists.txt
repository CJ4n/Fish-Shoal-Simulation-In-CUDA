set(APP_NAME fishShoal)
cmake_minimum_required(VERSION 3.16)

include_directories(include includeNvidia) 

add_compile_options(-Wall -Wextra)
add_link_options(-L/usr/lib/nvidia-compute-utils-510 -lGL -lGLU -lglut -Wall -Wextra)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
project(FishShoal LANGUAGES CXX CUDA)

file(GLOB cuda_sources src/*.cu)

add_executable(${APP_NAME} ${cuda_sources} )

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

target_link_libraries(${APP_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )

set_target_properties(${APP_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_features(${APP_NAME} PUBLIC cxx_std_17)
target_compile_features(${APP_NAME} PUBLIC cuda_std_17)
