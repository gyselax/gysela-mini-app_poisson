// SPDX-License-Identifier: MIT
#include "circular_to_cartesian.hpp"
#include "czarny_to_cartesian.hpp"
#include "discrete_poloidal_cs_spline_mapping.hpp"
#include "paraconfpp.hpp"
#include "poisson_init.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"

using DiscreteMapping = DiscretePoloidalCSSplineMapping<
        X,
        Y,
        SplineRThetaEvaluatorConstBound_host,
        R,
        Theta,
        Kokkos::HostSpace>;

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>>
initialise_polar_fem_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        SplineRThetaBuilder const& builder,
        SplineRThetaEvaluatorConstBound const& evaluator)
{
    // Parse optional arguments
    long int max_iter;
    double res_tol;
    int batch_solver_logger;
    long int preconditioner_max_block_size;

    PC_status_t max_iter_status = PC_int(PC_get(conf_gyselalibxx, ".Poisson.max_iter"), &max_iter);
    PC_status_t res_tol_status = PC_double(PC_get(conf_gyselalibxx, ".Poisson.res_tol"), &res_tol);
    PC_status_t batch_solver_logger_status = PC_bool(
            PC_get(conf_gyselalibxx, ".Poisson.batch_solver_logger"),
            &batch_solver_logger);
    PC_status_t preconditioner_max_block_size_status = PC_int(
            PC_get(conf_gyselalibxx, ".Poisson.preconditioner_max_block_size"),
            &preconditioner_max_block_size);

    std::optional<int> input_max_iter(
            max_iter_status == PC_OK ? std::optional<int>(max_iter) : std::nullopt);
    std::optional<double> input_res_tol(
            res_tol_status == PC_OK ? std::optional<double>(res_tol) : std::nullopt);
    std::optional<bool> input_batch_solver_logger(
            batch_solver_logger_status == PC_OK ? std::optional<bool>(batch_solver_logger)
                                                : std::nullopt);
    std::optional<int> input_preconditioner_max_block_size(
            preconditioner_max_block_size_status == PC_OK
                    ? std::optional<int>(preconditioner_max_block_size)
                    : std::nullopt);

    // Initialise Polar splines
    ddc::init_discrete_space<PolarBSplinesRTheta>(discrete_mapping);

    // Create pointer
    return std::make_unique<PolarSplineFEMPoissonLikeSolver<
            GridR,
            GridTheta,
            PolarBSplinesRTheta,
            SplineRThetaBuilder_host,
            SplineRThetaEvaluatorConstBound_host,
            DiscreteMapping>>(
            discrete_mapping,
            builder,
            evaluator,
            input_max_iter,
            input_res_tol,
            input_batch_solver_logger,
            input_preconditioner_max_block_size);
}

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>>
initialise_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        SplineRThetaBuilder const& builder,
        SplineRThetaEvaluatorConstBound const& evaluator)
{
    std::string algorithm(PCpp_string(conf_gyselalibxx, ".Poisson.algorithm"));
    if (algorithm == "PolarFEM") {
        return initialise_polar_fem_solver(conf_gyselalibxx, discrete_mapping, builder, evaluator);
    } else if (algorithm == "GMGPolar") {
        throw std::runtime_error("GMGPolar is not yet available");
    } else if (algorithm == "HyTeg") {
        throw std::runtime_error("HyTeg is not yet available");
    } else {
        throw std::runtime_error("Algorithm not recognised. Should be one of [PolarFEM, GMGPolar, "
                                 "HyTeg]");
    }
}
