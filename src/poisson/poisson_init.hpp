// SPDX-License-Identifier: MIT
#include <memory>

#include <ddc/ddc.hpp>

#include <paraconf.h>

#include "discrete_poloidal_cs_spline_mapping_builder.hpp"
#include "geometry_r_theta.hpp"
#include "ipolar_poisson_like_solver.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"
#include "spline_definitions_r_theta.hpp"

struct A
{
};
struct B
{
};

struct GridA : UniformGridBase<A>
{
};

struct GridB : UniformGridBase<B>
{
};

namespace {

using DiscreteMappingBuilder
        = DiscretePoloidalCSSplineMappingBuilder<X, Y, SplineInterpolatorRThetaConst>;
using DiscreteMapping = typename DiscreteMappingBuilder::MappingType;
using MetricTensor = MetricTensorEvaluator<DiscreteMapping>;

using IdxRangeBatch = IdxRange<GridA, GridB>;
using IdxBatch = Idx<GridA, GridB>;
using IdxRangeBatched = IdxRange<GridA, GridB, GridR, GridTheta>;
using IdxBatched = Idx<GridA, GridB, GridR, GridTheta>;

using DFieldMemBatched = DFieldMem<IdxRangeBatched>;
using DFieldBatched = DField<IdxRangeBatched>;

constexpr double res_tol = 1e-13;


} // namespace

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeBatched>>
initialise_batched_polar_fem_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscretePoloidalCSSplineMapping<X, Y, SplineRThetaEvaluatorConstBound, R, Theta> const&
                discrete_mapping,
        SplineInterpolatorRThetaConst const& interpolator,
        IdxRangeBatched const idx_range_batched);

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeRTheta>> initialise_gmgpolar_solver(
        PC_tree_t const& conf_gyselalibxx,
        DiscreteMapping const& discrete_mapping,
        SplineInterpolatorRThetaConst const& interpolator);
