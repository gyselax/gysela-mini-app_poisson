
add_executable(gys-poisson
    gys_poisson.cpp
)
target_link_libraries(gys-poisson
    PUBLIC
        DDC::core
        paraconf::paraconf
        gslx::geometry_RTheta
        gslx::paraconfpp
        gslx::pde_solvers
        gslx::poisson_RTheta
        gslx::utils
        test_cases
)
target_compile_definitions(gys-poisson PUBLIC -DCIRCULAR_MAPPING)
target_compile_definitions(gys-poisson PUBLIC -DCURVILINEAR_SOLUTION)
