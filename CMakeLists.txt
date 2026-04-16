cmake_minimum_required(VERSION 3.15)
project(simulation)


set(CMAKE_CXX_STANDARD 17)


set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed")
#fing eigen library
find_package (Eigen3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

message(STATUS "Eigen3 : " ${EIGEN3_INCLUDE_DIR}/Eigen)
message(STATUS "current : " ${CMAKE_CURRENT_SOURCE_DIR})


add_definitions(-DMJSIMULATE_STATIC) # use static =,solve the "error LNK2001/LNK2019" in windows

find_package (X11)
find_library(GLFW libglfw.so.3 HINTS ${CMAKE_SOURCE_DIR}/mujoco-3.6.0/bin)
# set MUJOCO root
set(MUJOCO_HOME ${CMAKE_SOURCE_DIR}/mujoco-3.6.0/)
message(STATUS "MUJOCO_HOME: " ${MUJOCO_HOME})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib
)

#$ENV{EIGEN3_INCLUDE_DIR}

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco-3.6.0/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco-3.6.0/simulate
  $ENV{EIGEN3_INCLUDE_DIR}/Eigen/
  ${GLEW_INCLUDE_DIRS}
    /usr/local/include/rbdl
    /usr/local/lib
)

add_subdirectory(src/control)
add_subdirectory(src/log)
add_subdirectory(src/planning)
add_subdirectory(src/robot)
add_subdirectory(src/utils)
add_subdirectory(src/policy)
add_subdirectory(src/observation)

set(COM_SRC
    src/main.cc
    ${MUJOCO_HOME}/simulate/glfw_dispatch.cc
    ${MUJOCO_HOME}/simulate/glfw_adapter.cc
    ${MUJOCO_HOME}/simulate/simulate.cc
    ${MUJOCO_HOME}/simulate/platform_ui_adapter.cc    
    
        
        # src/controller.cpp
        # src/robotmodel.cpp
        # src/trajectory.cpp
        )

set(COM_LIB 
${MUJOCO_HOME}/lib/libmujoco.so
${MUJOCO_HOME}/lib/libmujoco.so.3.6.0
${MUJOCO_HOME}/bin/mujoco_plugin/libelasticity.so
${GLFW}
)
add_executable(run ${COM_SRC})
#add_executable(${PROJECT_NAME} simulate/main.cc)
target_link_libraries(run 
    control # control has dependency on robot, planning
	${COM_LIB} 
	${CMAKE_DL_LIBS} 
	${X11_LIBRARIES}
	-lrbdl
    	-lrbdl_urdfreader
    	-lpthread
    	OpenGL::OpenGL
	OpenGL::EGL
	GLEW::GLEW
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
    -lonnxruntime
    policy
    observation
)

add_custom_command(TARGET run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${MUJOCO_HOME}/bin/mujoco_plugin
    $<TARGET_FILE_DIR:run>/mujoco_plugin
)