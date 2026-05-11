
set(MAPPING_LIST CIRCULAR CZARNY)
set(SOLUTION_LIST CURVILINEAR CARTESIAN)

foreach (MAPPING IN LISTS MAPPING_LIST)
    foreach (SOLUTION IN LISTS SOLUTION_LIST)
        add_executable(gys-poisson_${MAPPING}_${SOLUTION}
            gys_poisson.cpp
        )
        target_link_libraries(gys-poisson_${MAPPING}_${SOLUTION}
            PUBLIC
                DDC::core
                paraconf::paraconf
                gslx::geometry_RTheta
                gslx::paraconfpp
                gslx::pde_solvers
                gslx::poisson_RTheta
                gslx::utils
                poisson_init
                test_cases
        )
        target_compile_definitions(gys-poisson_${MAPPING}_${SOLUTION} PUBLIC -D${MAPPING}_MAPPING)
        target_compile_definitions(gys-poisson_${MAPPING}_${SOLUTION} PUBLIC -D${SOLUTION}_SOLUTION)
    endforeach()
endforeach()
