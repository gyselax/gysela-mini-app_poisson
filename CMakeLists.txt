# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(gysela-mini-app_poisson CXX)

###############################################################################################
#                           List of options for Gyselalib++ library
###############################################################################################
set(GYSELALIBXX_BUILD_TESTING OFF)
set(GYSELALIBXX_BUILD_SIMULATIONS OFF)
set(GYSELALIBXX_BUILD_DOCUMENTATION OFF)
set(GYSELALIBXX_INCLUDE_TESTING_DEPENDENCIES ON)


###############################################################################################
#                                   List of options
###############################################################################################

###############################################################################################
#                              Build libraries and executables
###############################################################################################

set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

## Link to Gyselalibxx++ library
add_subdirectory(gyselalibxx)

## Change the C++ flags to the Gysela-X++ specific ones
set(CMAKE_CXX_FLAGS "${GYSELAX_DEFAULT_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

## Get the main code in `src/`
add_subdirectory(src/)

## Build the simulations (applications mains) in `simulations/`
#add_subdirectory(simulations/)


