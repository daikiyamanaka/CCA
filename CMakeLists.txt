project(SimpleCCA)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
aux_source_directory(src SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

######################
# boost
######################
#include_directories("/usr/local/include")
#link_directories("/usr/local/lib")
#target_link_libraries(${PROJECT_NAME} boost_filesystem-mt boost_thread-mt boost_program_options-mt boost_system-mt)
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
set(Boost_INCLUDE_DIR /usr/include)
set(Boost_LIBRARY_DIR /usr/lib)
set(BOOST_LIBRARYDIR /usr/lib)
link_directories(/usr/lib)
get_directory_property(OUT_VAR LINK_DIRECTORIES)
message(STATUS "LINK DIR: ${OUT_VAR}")
endif()


FIND_PACKAGE(Boost COMPONENTS filesystem thread program_options system date_time REQUIRED)
if (Boost_FOUND)
  message(STATUS "Found 'boost library'")
  include_directories(${Boost_INCLUDE_DIR})
  link_directories(${Boost_LIBRARY_DIRS})
  message(STATUS " boost lib dir:${Boost_LIBRARY_DIRS}")
  message(STATUS " boost include dir:${Boost_INCLUDE_DIR}")
  message(STATUS " libs: ${Boost_LIBRARIES}" )

  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
  if(APPLE)
  #target_link_libraries(${PROJECT_NAME} boost_filesystem-mt boost_thread-mt boost_program_options-mt boost_system-mt)
  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
  elseif(UNIX)
  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
  endif()

else()
  message(FATAL_ERROR "Couldn't find Boost library.")
endif()

##########################
# opencv
##########################
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/local/opencv/share/OpenCV/OpenCVConfig.cmake")
set(OpenCV_DIR "/usr/local/opencv/share/OpenCV/OpenCVConfig.cmake"
  CACHE PATH "The path where OpenCVConfig.cmake is placed")
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
  message(STATUS "Found 'opencv library' ${OpenCV_VERSION}")
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIB_DIR})
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
else()
  message( SEND_ERROR "You can't do that" )
endif()

##########
#Eigen
##########
include_directories("/Users/daikiyamanaka/Program/eigen-3.1.3")

#######
#ft
######
#set(LIB_FT_DIR "${PROJECT_SOURCE_DIR}/ft")
#add_subdirectory(${LIB_FT_DIR})

#get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
#message("inc_dirs = ${inc_dirs}")
#target_link_libraries(${PROJECT_NAME} ft)

#target_link_libraries(${PROJECT_NAME} /Users/daikiyamanaka/Dropbox/abeja/program/genderClassification/lib/libft.a)

######################                                                                                                                                           
#OpenMP                                                                                                                                                           
##################### 
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

#add_subdirectory(${PROJECT_SOURCE_DIR}/src/liblinear)
#target_link_libraries(${PROJECT_NAME} linear)
