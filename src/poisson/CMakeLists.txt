add_library(poisson_init STATIC)

target_sources(poisson_init PRIVATE poisson_init.cpp PUBLIC FILE_SET HEADERS FILES poisson_init.hpp)

target_link_libraries(
    poisson_init
    PUBLIC
        paraconf::paraconf
        DDC::core
        DDC::splines
        gyselalibxx::geometry_RTheta
        gyselalibxx::paraconfpp
        gyselalibxx::pde_solvers
        gyselalibxx::utils
)
