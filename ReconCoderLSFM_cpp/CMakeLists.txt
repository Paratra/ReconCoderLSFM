###  20230120
#__author__: Ming Song， UGA, Kner Lab
#__function__: SIM 2D reconstruction

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
cmake_minimum_required(VERSION 2.8)
project( sim_2d_reconstruction )



#######  find_packages
### OpenCV ###
find_package( OpenCV REQUIRED )
message( ${OpenCV_VERSION} )

find_package(TIFF REQUIRED)
#message( ${TIFF_VERSION} )


# add_library(ReconCoderLSFM src/ReconCoderLSFM.cpp)
add_library(Utility src/Utility.cpp)
# add_library(tools tools.cpp)

######## include directories
include_directories(
  include
  ${TIFF_INCLUDE_DIR}
)

add_executable( ReconCoderLSFM
  src/ReconCoderLSFM.cpp
  src/Utility.cpp
  include/ReconCoderLSFM.h
  include/Utility.h
)

target_link_libraries(
    ReconCoderLSFM ${TIFF_LIBRARIES} ${OpenCV_LIBS} Utility
)



target_link_libraries(
  Utility ${OpenCV_LIBS}
)

# add_executable( Utility
#   src/Utility.cpp
#   include/Utility.h
# )

