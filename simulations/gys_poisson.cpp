// SPDX-License-Identifier: MIT
#include <paraconf.h>
#include <pdi.h>

#include <chrono>
#include <ddc/ddc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "cartesian_solution.hpp"
#include "circular_to_cartesian.hpp"
#include "curvilinear_solution.hpp"
#include "czarny_to_cartesian.hpp"
#include "ddc_alias_inline_functions.hpp"
#include "geometry_r_theta.hpp"
#include "input.hpp"
#include "ipolar_poisson_like_solver.hpp"
#include "l_norm_tools.hpp"
#include "math_tools.hpp"
#include "paraconfpp.hpp"
#include "params.yaml.hpp"
#include "rhs.hpp"
#include "spline_definitions_r_theta.hpp"

namespace fs = std::filesystem;

#if defined(CIRCULAR_MAPPING)
using AnalyticalMapping = CircularToCartesian<R, Theta, X, Y>;
#elif defined(CZARNY_MAPPING)
using AnalyticalMapping = CzarnyToCartesian<R, Theta, X, Y>;
#endif

#if defined(CURVILINEAR_SOLUTION)
using Solution = CurvilinearSolution<AnalyticalMapping>;
#elif defined(CARTESIAN_SOLUTION)
using Solution = CartesianSolution<AnalyticalMapping>;
#endif

int main(int argc, char** argv) {
    ::Kokkos::ScopeGuard kokkos_scope(argc, argv);
    ::ddc::ScopeGuard ddc_scope(argc, argv);

    // Parse command line arguments
    PC_tree_t conf_gyselalibxx;
    if (argc == 2) {
        conf_gyselalibxx = PC_parse_path(fs::path(argv[1]).c_str());
    } else if (argc == 3) {
        if (argv[1] == std::string_view("--dump-config")) {
            std::fstream file(argv[2], std::fstream::out);
            file << params_yaml;
            return EXIT_SUCCESS;
        }
    } else {
        std::cerr << "usage: " << argv[0]
                  << " [--dump-config] <config_file.yml>" << std::endl;
        return EXIT_FAILURE;
    }
    PC_errhandler(PC_NULL_HANDLER);

    // -------------------------------------------------------------
    //                        Setup
    // -------------------------------------------------------------
    std::chrono::time_point<std::chrono::system_clock> start_time =
        std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> end_time;

    CoordR r_min(PCpp_double(conf_gyselalibxx, ".SplineMesh.r_min"));
    CoordR r_max(PCpp_double(conf_gyselalibxx, ".SplineMesh.r_max"));
    IdxStepR r_ncells(PCpp_int(conf_gyselalibxx, ".SplineMesh.r_ncells"));

    CoordTheta theta_min(
        PCpp_double(conf_gyselalibxx, ".SplineMesh.theta_min"));
    CoordTheta theta_max(
        PCpp_double(conf_gyselalibxx, ".SplineMesh.theta_max"));
    IdxStepTheta theta_ncells(
        PCpp_int(conf_gyselalibxx, ".SplineMesh.theta_ncells"));

    std::vector<CoordR> breakpoints_r =
        build_uniform_break_points(r_min, r_max, r_ncells);
    std::vector<CoordTheta> breakpoints_theta =
        build_uniform_break_points(theta_min, theta_max, theta_ncells);

    ddc::init_discrete_space<BSplinesR>(breakpoints_r);
    ddc::init_discrete_space<GridR>(
        SplineInterpPointsR::template get_sampling<GridR>());

    ddc::init_discrete_space<BSplinesTheta>(breakpoints_theta);
    ddc::init_discrete_space<GridTheta>(
        SplineInterpPointsTheta::template get_sampling<GridTheta>());

    IdxRangeR idxrange_r = SplineInterpPointsR::template get_domain<GridR>();
    IdxRangeTheta idxrange_theta =
        SplineInterpPointsTheta::template get_domain<GridTheta>();
    IdxRangeRTheta idx_range(idxrange_r, idxrange_theta);

    // setup mapping
    AnalyticalMapping mapping;

    host_t<DFieldMemRTheta> coeff_alpha_alloc(
        idx_range);  // values of the coefficient alpha
    host_t<DFieldMemRTheta> coeff_beta_alloc(idx_range);

    host_t<DFieldRTheta> coeff_alpha =
        get_field(coeff_alpha_alloc);  // values of the coefficient alpha
    host_t<DFieldRTheta> coeff_beta = get_field(coeff_beta_alloc);

    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(), idx_range,
        KOKKOS_LAMBDA(IdxRTheta const irtheta) {
            coeff_alpha(irtheta) = Kokkos::exp(-Kokkos::tanh(
                (ddc::coordinate(ddc::select<GridR>(irtheta)) - 0.7) / 0.05));
            coeff_beta(irtheta) = 1.0 / coeff_alpha(irtheta);
        });

    end_time = std::chrono::system_clock::now();
    std::cout << "Setup time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_time - start_time)
                     .count()
              << "ms" << std::endl;
    start_time = std::chrono::system_clock::now();

    // -------------------------------------------------------------
    //                    Initialise Poisson
    // -------------------------------------------------------------
    std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta,
                                            Kokkos::HostSpace>>
        solver;
    // TODO: Choose solver

    solver->update_coefficients(get_const_field(coeff_alpha),
                                get_const_field(coeff_beta));

    end_time = std::chrono::system_clock::now();
    std::cout << "Poisson initialisation time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_time - start_time)
                     .count()
              << "ms" << std::endl;

    // -------------------------------------------------------------
    //                 Initialise Poisson input
    // -------------------------------------------------------------
    double major_radius = 6.1;
    double vertical_offset = 0.3;
    Solution lhs(mapping, major_radius, vertical_offset);

    ManufacturedRHS<Solution> rhs_calculator(mapping);

    host_t<DFieldMemRTheta> result_alloc(idx_range);
    host_t<DFieldRTheta> result = get_field(result_alloc);

    host_t<DFieldMemRTheta> rhs_alloc(idx_range);
    host_t<DFieldRTheta> rhs = get_field(result_alloc);

    ddc::host_for_each(idx_range, [&](IdxRTheta idx) {
        rhs(idx) = rhs_calculator(ddc::coordinate(idx));
    });

    // -------------------------------------------------------------
    //                 Solve Poisson equation
    // -------------------------------------------------------------
    start_time = std::chrono::system_clock::now();
    (*solver)(result, get_const_field(rhs));
    end_time = std::chrono::system_clock::now();
    std::cout << "Solver time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_time - start_time)
                     .count()
              << "ms" << std::endl;

    // -------------------------------------------------------------
    //                 Check error
    // -------------------------------------------------------------
    double max_err = error_norm_inf(
        Kokkos::DefaultHostExecutionSpace(), get_const_field(result),
        [&](IdxRTheta const irtheta) { return lhs(ddc::coordinate(irtheta)); });
    std::cout << "Max error : " << max_err << std::endl;

    PC_tree_destroy(&conf_gyselalibxx);
    return 0;
}
