cmake_minimum_required(VERSION 3.10)
project (dicgpu)

find_package( OpenCV REQUIRED )

find_package(OpenCL REQUIRED)

add_executable(dicgpu 
	src/main.cpp 
	src/main.h 
	
	src/clio.cpp 
	src/clio.h 
	
	src/SimpleLogger.cpp
	src/SimpleLogger.h

	src/SimpleConfig.cpp 
	src/SimpleConfig.h 
	
	src/imageio.cpp 
	src/imageio.h 
	
	src/common.cpp
	src/common.h)

target_link_libraries( dicgpu ${OpenCV_LIBS} OpenCL::OpenCL)
