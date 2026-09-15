# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(gysela-mini-app_poisson CXX)

###############################################################################################
#                           List of options for Gyselalib++ library
###############################################################################################
set(GYSELALIBXX_BUILD_TESTING OFF)
set(GYSELALIBXX_BUILD_SIMULATIONS OFF)
set(GYSELALIBXX_BUILD_DOCUMENTATION OFF)


###############################################################################################
#                                   List of options
###############################################################################################

###############################################################################################
#                              Build libraries and executables
###############################################################################################

set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

## Link to Gyselalibxx++ library
add_subdirectory(gyselalibxx EXCLUDE_FROM_ALL)

## Get the main code in `src/`
add_subdirectory(src/)

## Build the simulations (applications mains) in `simulations/`
add_subdirectory(simulations/)


