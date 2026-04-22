# SPDX-License-Identifier: MIT

add_library(test_cases STATIC
    rhs_cartesian_solution_circular_mapping.cpp
    rhs_curvilinear_solution_circular_mapping.cpp
    rhs_cartesian_solution_czarny_mapping.cpp
    rhs_curvilinear_solution_czarny_mapping.cpp
)

target_include_directories(test_cases
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(test_cases
    PUBLIC
        DDC::core

        gslx::geometry_RTheta
        gslx::utils
)


