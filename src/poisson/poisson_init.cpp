// SPDX-License-Identifier: MIT
#include "circular_to_cartesian.hpp"
#include "czarny_to_cartesian.hpp"
#include "discrete_poloidal_cs_spline_mapping.hpp"
#include "hyteg_poisson_like_solver.hpp"
#include "paraconfpp.hpp"
#include "poisson_init.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"

#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/p2functionspace/P2Function.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearProlongation.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearRestriction.hpp"
#include "hyteg/gridtransferoperators/P2toP2QuadraticProlongation.hpp"
#include "hyteg/gridtransferoperators/P2toP2QuadraticRestriction.hpp"
#include "hyteg_operators/operators/div_alpha_grad_plus_beta_mass/P1ElementwiseDivAlphaGradPlusBetaMassP1CoeffsParametricP1Map.hpp"
#include "hyteg_operators/operators/div_alpha_grad_plus_beta_mass/P2ElementwiseDivAlphaGradPlusBetaMassP2CoeffsParametricP2Map.hpp"
#include "hyteg_operators/operators/mass/P1ElementwiseMassParametricP1Map.hpp"
#include "hyteg_operators/operators/mass/P2ElementwiseMassParametricP2Map.hpp"

using DiscreteMapping
        = DiscretePoloidalCSSplineMapping<X, Y, SplineRThetaEvaluatorConstBound, R, Theta>;

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>>
initialise_polar_fem_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        SplineInterpolatorRThetaConst const& interpolator)
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
            SplineInterpolatorRThetaConst,
            DiscreteMapping>>(
            discrete_mapping,
            interpolator,
            input_max_iter,
            input_res_tol,
            input_batch_solver_logger,
            input_preconditioner_max_block_size);
}

template <
        class FunctionType,
        class OperatorType,
        class MassType,
        class RestrictionType,
        class ProlongationType,
        class MicroMeshFuncType>
std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>>
initialise_hyteg_solver_impl(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        IdxRangeRTheta idx_range)
{
    // Parse optional arguments
    long int num_levels;
    long int cg_max_iter;
    double cg_tol;

    PC_status_t num_levels_status
            = PC_int(PC_get(conf_gyselalibxx, ".Poisson.num_levels"), &num_levels);
    PC_status_t cg_max_iter_status
            = PC_int(PC_get(conf_gyselalibxx, ".Poisson.cg_max_iter"), &cg_max_iter);
    PC_status_t cg_tol_status = PC_double(PC_get(conf_gyselalibxx, ".Poisson.cg_tol"), &cg_tol);

    std::optional<int> input_num_levels(
            num_levels_status == PC_OK ? std::optional<int>(num_levels) : std::nullopt);
    std::optional<int> input_cg_max_iter(
            cg_max_iter_status == PC_OK ? std::optional<int>(cg_max_iter) : std::nullopt);
    std::optional<double> input_cg_tol(
            cg_tol_status == PC_OK ? std::optional<double>(cg_tol) : std::nullopt);

    return std::make_unique<HyTegPoissonLikeSolver<
            DiscreteMapping,
            GridR,
            GridTheta,
            FunctionType,
            OperatorType,
            MassType,
            RestrictionType,
            ProlongationType,
            MicroMeshFuncType>>(
            discrete_mapping,
            idx_range,
            input_num_levels,
            input_cg_max_iter,
            input_cg_tol);
}

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>> initialise_hyteg_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        IdxRangeRTheta idx_range)
{
    char* c_polynomial_degree = nullptr;
    PC_status_t status = PC_string(
            PC_get(conf_gyselalibxx, ".Poisson.polynomial_degree"),
            &c_polynomial_degree);
    std::string const polynomial_degree(status == PC_OK ? c_polynomial_degree : "P2");
    if (c_polynomial_degree != nullptr) {
        free(c_polynomial_degree);
    }

    using real_t = walberla::real_t;
    if (polynomial_degree == "P1") {
        return initialise_hyteg_solver_impl<
                hyteg::P1Function<real_t>,
                hyteg::operatorgeneration::P1ElementwiseDivAlphaGradPlusBetaMassP1CoeffsParametricP1Map,
                hyteg::operatorgeneration::P1ElementwiseMassParametricP1Map,
                hyteg::P1toP1LinearRestriction<real_t>,
                hyteg::P1toP1LinearProlongation<real_t>,
                hyteg::P1VectorFunction<real_t>>(conf_gyselalibxx, discrete_mapping, idx_range);
    } else if (polynomial_degree == "P2") {
        return initialise_hyteg_solver_impl<
                hyteg::P2Function<real_t>,
                hyteg::operatorgeneration::P2ElementwiseDivAlphaGradPlusBetaMassP2CoeffsParametricP2Map,
                hyteg::operatorgeneration::P2ElementwiseMassParametricP2Map,
                hyteg::P2toP2QuadraticRestriction,
                hyteg::P2toP2QuadraticProlongation,
                hyteg::P2VectorFunction<real_t>>(conf_gyselalibxx, discrete_mapping, idx_range);
    } else {
        throw std::runtime_error(
                "Poisson.polynomial_degree not recognised. Should be one of [P1, P2]");
    }
}

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>> initialise_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        SplineInterpolatorRThetaConst const& interpolator)
{
    std::string algorithm(PCpp_string(conf_gyselalibxx, ".Poisson.algorithm"));
    if (algorithm == "PolarFEM") {
        return initialise_polar_fem_solver(conf_gyselalibxx, discrete_mapping, interpolator);
    } else if (algorithm == "GMGPolar") {
        throw std::runtime_error("GMGPolar is not yet available");
    } else if (algorithm == "HyTeg") {
        IdxRangeRTheta const idx_range = get_spline_idx_range(interpolator.get_builder());
        return initialise_hyteg_solver(conf_gyselalibxx, discrete_mapping, idx_range);
    } else {
        throw std::runtime_error("Algorithm not recognised. Should be one of [PolarFEM, GMGPolar, "
                                 "HyTeg]");
    }
}
