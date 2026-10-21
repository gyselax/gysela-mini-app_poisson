

add_library(poisson_init STATIC
    poisson_init.cpp
)
target_include_directories(poisson_init
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(poisson_init
    PUBLIC
        paraconf::paraconf
        DDC::core
        DDC::splines
        gyselalibxx::geometry_RTheta
        gyselalibxx::paraconfpp
        gyselalibxx::pde_solvers
        gyselalibxx::utils
)

