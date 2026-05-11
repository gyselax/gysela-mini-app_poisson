// SPDX-License-Identifier: MIT
#include <memory>

#include <ddc/ddc.hpp>

#include <paraconf.h>

#include "geometry_r_theta.hpp"
#include "ipolar_poisson_like_solver.hpp"
#include "spline_definitions_r_theta.hpp"

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta, Kokkos::HostSpace>>
initialise_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscretePoloidalCSSplineMapping<
                X,
                Y,
                SplineRThetaEvaluatorConstBound_host,
                R,
                Theta,
                Kokkos::HostSpace> const& discrete_mapping,
        SplineRThetaBuilder_host const& builder,
        SplineRThetaEvaluatorConstBound_host const& evaluator);
