################## OPENCV #####################
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
###############################################

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/src/slic.hpp
	${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/matrix.hpp
    ${PROJECT_SOURCE_DIR}/src/util.hpp
    ${PROJECT_SOURCE_DIR}/src/util.cpp
)

###################################################################################################
# - library targets -------------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    ${PROTOBUF_LIB}
    ${PROTOBUF_LIBRARIES}
)