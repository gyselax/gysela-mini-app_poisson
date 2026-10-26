// SPDX-License-Identifier: MIT
#include "circular_to_cartesian.hpp"
#include "czarny_to_cartesian.hpp"
#include "discrete_poloidal_cs_spline_mapping.hpp"
#include "gmg_polar_poisson_like_solver.hpp"
#include "paraconfpp.hpp"
#include "poisson_init.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"



std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeBatched>>
initialise_batched_polar_fem_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        SplineInterpolatorRThetaConst const& interpolator,
        IdxRangeBatched const idx_range_batched)
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
            batch_solver_logger_status == PC_OK ? std::optional<bool>(true) : std::nullopt);
    std::optional<int> input_preconditioner_max_block_size(
            preconditioner_max_block_size_status == PC_OK
                    ? std::optional<int>(preconditioner_max_block_size)
                    : std::nullopt);

    // Initialise Polar splines
    ddc::init_discrete_space<PolarBSplinesRTheta>(discrete_mapping);
    MetricTensor const metric_tensor(discrete_mapping);


    // Create pointer
    return std::make_unique<PolarSplineFEMPoissonLikeSolver<
            GridR,
            GridTheta,
            PolarBSplinesRTheta,
            SplineInterpolatorRThetaConst,
            DiscreteMapping,
            MetricTensor,
            IdxRangeBatched>>(
            discrete_mapping,
            metric_tensor,
            interpolator,
            idx_range_batched,
            input_max_iter,
            input_res_tol,
            input_batch_solver_logger,
            input_preconditioner_max_block_size);
}

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>> initialise_gmgpolar_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        SplineInterpolatorRThetaConst const& interpolator)
{
    // Parse optional arguments
    int with_extrapolation;
    long int max_iter;
    double abs_tol;
    double rel_tol;

    PC_status_t with_extrapolation_status
            = PC_bool(PC_get(conf_gyselalibxx, ".Poisson.with_extrapolation"), &with_extrapolation);
    PC_status_t max_iter_status = PC_int(PC_get(conf_gyselalibxx, ".Poisson.max_iter"), &max_iter);
    PC_status_t abs_tol_status = PC_double(PC_get(conf_gyselalibxx, ".Poisson.abs_tol"), &abs_tol);
    PC_status_t rel_tol_status = PC_double(PC_get(conf_gyselalibxx, ".Poisson.rel_tol"), &rel_tol);

    gmgpolar::ExtrapolationType input_with_extrapolation;
    if (with_extrapolation_status == PC_OK) {
        input_with_extrapolation = with_extrapolation
                                           ? gmgpolar::ExtrapolationType::IMPLICIT_EXTRAPOLATION
                                           : gmgpolar::ExtrapolationType::NONE;
    } else {
        input_with_extrapolation = gmgpolar::ExtrapolationType::NONE;
    }
    std::optional<int> input_max_iter(
            max_iter_status == PC_OK ? std::optional<int>(max_iter) : std::nullopt);
    std::optional<double> input_abs_tol(
            abs_tol_status == PC_OK ? std::optional<double>(abs_tol) : std::nullopt);
    std::optional<double> input_rel_tol(
            rel_tol_status == PC_OK ? std::optional<double>(rel_tol) : std::nullopt);

    return std::make_unique<GMGPolarPoissonLikeSolver<
            DiscreteMapping,
            GridR,
            GridTheta,
            SplineInterpolatorRThetaConst>>(
            discrete_mapping,
            interpolator,
            input_with_extrapolation,
            input_max_iter,
            input_abs_tol,
            input_rel_tol);
}
