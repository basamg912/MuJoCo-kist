add_library(observation STATIC # observation 이름의 동적라이브러리 파일 생성
    observation.cpp
)

target_include_directories(observation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/mujoco-3.6.0/include
)

target_link_libraries(observation PUBLIC
    Eigen3::Eigen
)

