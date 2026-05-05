// SPDX-License-Identifier: MIT
#include "circular_to_cartesian.hpp"
#include "czarny_to_cartesian.hpp"
#include "discrete_poloidal_cs_spline_mapping.hpp"
#include "discrete_poloidal_cs_spline_mapping_builder.hpp"
#include "paraconfpp.hpp"
#include "poisson_init.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"

template <class MappingType>
std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta, Kokkos::HostSpace>>
initialise_polar_fem_solver(
        PC_tree_t const& conf_gyselalibxx,
        MappingType const& mapping,
        SplineRThetaBuilder_host const& builder,
        SplineRThetaEvaluatorConstBound_host const& evaluator)
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
    // TODO: Add constexpr if argument is DiscretePoloidalCSSplineMapping
    DiscretePoloidalCSSplineMappingBuilder<
            X,
            Y,
            SplineRThetaBuilder_host,
            SplineRThetaEvaluatorConstBound_host> const
            discrete_mapping_builder(
                    Kokkos::DefaultHostExecutionSpace(),
                    mapping,
                    builder,
                    evaluator);
    DiscretePoloidalCSSplineMapping const discrete_mapping = discrete_mapping_builder();

    ddc::init_discrete_space<PolarBSplinesRTheta>(discrete_mapping);

    // Create pointer
    return std::make_unique<PolarSplineFEMPoissonLikeSolver<
            GridR,
            GridTheta,
            PolarBSplinesRTheta,
            SplineRThetaBuilder_host,
            SplineRThetaEvaluatorConstBound_host,
            MappingType>>(
            mapping,
            builder,
            evaluator,
            input_max_iter,
            input_res_tol,
            input_batch_solver_logger,
            input_preconditioner_max_block_size);
}

template <class MappingType>
std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta, Kokkos::HostSpace>>
initialise_solver(
        PC_tree_t const& conf_gyselalibxx,
        MappingType const& mapping,
        SplineRThetaBuilder_host const& builder,
        SplineRThetaEvaluatorConstBound_host const& evaluator)
{
    std::string algorithm(PCpp_string(conf_gyselalibxx, ".Poisson.algorithm"));
    if (algorithm == "PolarFEM") {
        return initialise_polar_fem_solver(conf_gyselalibxx, mapping, builder, evaluator);
    } else if (algorithm == "GMGPolar") {
        throw std::runtime_error("GMGPolar is not yet available");
    } else if (algorithm == "HyTeg") {
        throw std::runtime_error("HyTeg is not yet available");
    } else {
        throw std::runtime_error("Algorithm not recognised. Should be one of [PolarFEM, GMGPolar, "
                                 "HyTeg]");
    }
}

template std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta, Kokkos::HostSpace>>
initialise_solver<CircularToCartesian<R, Theta, X, Y>>(
        PC_tree_t const& conf_gyselalibxx,
        CircularToCartesian<R, Theta, X, Y> const& mapping,
        SplineRThetaBuilder_host const& builder,
        SplineRThetaEvaluatorConstBound_host const& evaluator);

template std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta, Kokkos::HostSpace>>
initialise_solver<CzarnyToCartesian<R, Theta, X, Y>>(
        PC_tree_t const& conf_gyselalibxx,
        CzarnyToCartesian<R, Theta, X, Y> const& mapping,
        SplineRThetaBuilder_host const& builder,
        SplineRThetaEvaluatorConstBound_host const& evaluator);

// template std::unique_ptr<
//     IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta,
//     Kokkos::HostSpace>>
// initialise_solver<DiscretePoloidalCSSplineMapping<
//     X, Y, SplineRThetaEvaluatorConstBound_host>>(
//     PC_tree_t const& conf_gyselalibxx,
//     DiscretePoloidalCSSplineMapping<
//         X, Y, SplineRThetaEvaluatorConstBound_host> const& mapping,
//     SplineRThetaBuilder_host const& builder,
//     SplineRThetaEvaluatorConstBound_host const& evaluator);
