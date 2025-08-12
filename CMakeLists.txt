cmake_minimum_required(VERSION 3.15)
project(pytinybvh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# use local python .venv
if(WIN32)
    set(VENV_PYTHON "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
else()
    set(VENV_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()

# pybind11 include
execute_process(
        COMMAND ${VENV_PYTHON} -m pybind11 --includes
        OUTPUT_VARIABLE PYBIND11_INCLUDES_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Remove -I prefix from each path and add include_directories
string(REPLACE "-I" "" PYBIND11_INCLUDES_LIST ${PYBIND11_INCLUDES_RAW})
separate_arguments(PYBIND11_INCLUDES_LIST)

foreach(dir IN LISTS PYBIND11_INCLUDES_LIST)
    include_directories(${dir})
endforeach()

# Python headers
find_package(Python3 COMPONENTS Development REQUIRED)

include_directories(
        ${PYBIND11_INCLUDES}              # pybind11
        ${CMAKE_SOURCE_DIR}/deps          # tinybvh
        ${Python3_INCLUDE_DIRS}           # Python API
)

add_library(pytinybvh MODULE src/pytinybvh.cpp src/capabilities.h)

# avoid 'lib' prefix for Python extension modules on macOS
set_target_properties(pytinybvh PROPERTIES PREFIX "")

# Link to Python
target_link_libraries(pytinybvh PRIVATE Python3::Python)