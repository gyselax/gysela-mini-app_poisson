// SPDX-License-Identifier: MIT
#include <memory>

#include <ddc/ddc.hpp>

#include <paraconf.h>

#include "geometry_r_theta.hpp"
#include "ipolar_poisson_like_solver.hpp"
#include "spline_definitions_r_theta.hpp"

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>>
initialise_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscretePoloidalCSSplineMapping<
                X,
                Y,
                SplineRThetaEvaluatorConstBound,
                R,
                Theta> const& discrete_mapping,
        SplineRThetaBuilder const& builder,
        SplineRThetaEvaluatorConstBound const& evaluator);
