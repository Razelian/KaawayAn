cmake_minimum_required(VERSION 3.16)
# One sdkconfig per build directory, so xiao and devkit builds never share config.
set(SDKCONFIG ${CMAKE_BINARY_DIR}/sdkconfig)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(kaawayan_firmware)
