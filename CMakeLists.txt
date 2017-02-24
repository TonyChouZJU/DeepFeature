cmake_minimum_required(VERSION 2.8)
project(CaffeWorker)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

set(Caffe_DIR /home/zyb/py-faster-rcnn/caffe-fast-rcnn)
set(GNU_DIR /usr/lib/x86_64-linux-gnu/)
set(COSPEAK_DIR /mnt/exhdd/tomorning_dataset/wonderland/cv/Deep_retriver_worker)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(DP_SOURCE_FILES dp_main.cpp DeepFeatureExtractor.cpp DeepFeatureExtractor.hpp)
SET(CMAKE_BUILD_TYPE "Debug") 

FIND_LIBRARY(GLOG_LIB glog path ${GNU_DIR})
FIND_LIBRARY(BOOST_LIB boost_system path ${GNU_DIR})
FIND_LIBRARY(CAFFE_LIB caffe path ${Caffe_DIR}/lib)
FIND_LIBRARY(COSPEAK_LIB cospeak path  ${COSPEAK_DIR}/lib)

include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories(${Caffe_DIR}/include)

add_executable(DPWorker ${DP_SOURCE_FILES})
target_link_libraries(DPWorker ${OpenCV_LIBS} ${GLOG_LIB} ${CAFFE_LIB} ${BOOST_LIB} ${COSPEAK_LIB})
