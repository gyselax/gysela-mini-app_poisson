# SPDX-License-Identifier: MIT

add_library(test_cases STATIC)

target_sources(test_cases
    PRIVATE
      rhs_cartesian_solution_circular_mapping.cpp
      rhs_curvilinear_solution_circular_mapping.cpp
      rhs_cartesian_solution_czarny_mapping.cpp
      rhs_curvilinear_solution_czarny_mapping.cpp
    PUBLIC
        FILE_SET HEADERS
            FILES
                cartesian_solution.hpp
                curvilinear_solution.hpp
                i_solution.hpp
                rhs.hpp
)

target_link_libraries(test_cases
    PUBLIC
        DDC::core

        gyselalibxx::geometry_RTheta
        gyselalibxx::utils
)


