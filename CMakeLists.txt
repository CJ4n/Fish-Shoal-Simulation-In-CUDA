# set(CMAKE_C_COMPILER /usr/bin/gcc-9)
# set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
set(APP_NAME fishShoal)
cmake_minimum_required(VERSION 3.16)

include_directories(include includeNvidia) 

add_compile_options(-g -G)
add_link_options(-L/usr/lib/nvidia-compute-utils-510 -lGL -lGLU -lglut )

project(CudaApp LANGUAGES CXX CUDA)

file(GLOB cuda_sources "src/*.cu" )

add_executable(${APP_NAME} ${cuda_sources} )

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

target_link_libraries(${APP_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )

# find_package(OpenGL)
# find_package(GLUT)
# target_link_libraries(${APP_NAME} OpenGL::GL GLUT::GLUT)
set_target_properties(${APP_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_features(${APP_NAME} PUBLIC cxx_std_11)
target_compile_features(${APP_NAME} PUBLIC cuda_std_11)
