// SPDX-License-Identifier: MIT
#include <memory>

#include <ddc/ddc.hpp>

#include <paraconf.h>

#include "discrete_poloidal_cs_spline_mapping_builder.hpp"
#include "geometry_r_theta.hpp"
#include "ipolar_poisson_like_solver.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"
#include "spline_definitions_r_theta.hpp"

struct A {};
struct B {};

struct GridA : UniformGridBase<A> {};

struct GridB : UniformGridBase<B> {};

namespace {

using DiscreteMappingBuilder =
    DiscretePoloidalCSSplineMappingBuilder<X, Y, SplineInterpolatorRThetaConst>;
using DiscreteMapping = typename DiscreteMappingBuilder::MappingType;

using IdxRangeBatch = IdxRange<GridA, GridB>;
using IdxBatch = Idx<GridA, GridB>;
using IdxRangeBatched = IdxRange<GridA, GridB, GridR, GridTheta>;
using IdxBatched = Idx<GridA, GridB, GridR, GridTheta>;

using DFieldMemBatched = DFieldMem<IdxRangeBatched>;
using DFieldBatched = DField<IdxRangeBatched>;

using PoissonSolver =
    PolarSplineFEMPoissonLikeSolver<GridR, GridTheta, PolarBSplinesRTheta,
                                    SplineInterpolatorRTheta, DiscreteMapping>;

using BatchedPoissonSolver =
    PolarSplineFEMPoissonLikeSolver<GridR, GridTheta, PolarBSplinesRTheta,
                                    SplineInterpolatorRTheta, DiscreteMapping,
                                    IdxRangeBatched>;

constexpr double res_tol = 1e-13;

KOKKOS_FUNCTION double alpha_func(double r, double a) {
  return (1.0 + 0.5 * a) * Kokkos::exp(-Kokkos::tanh((r - 0.7) / 0.05));
}

KOKKOS_FUNCTION double beta_func(double r, double a, double b) {
  return (1.0 + b) / alpha_func(r, a);
}

KOKKOS_FUNCTION double rho_func(double r, double theta, double a, double b) {
  return (1.0 + a) * (1.0 - r * r) *
         (1.0 + 0.3 * (b + 1.0) * r * Kokkos::cos(theta));
}

/// The RHS of the equation for a given batch index.
class BatchedRHS {
public:
  KOKKOS_FUNCTION double operator()(IdxBatch idx_batch,
                                    CoordRTheta const &coord) const {
    double const a = ddc::coordinate(Idx<GridA>(idx_batch));
    double const b = ddc::coordinate(Idx<GridB>(idx_batch));
    return rho_func(ddc::get<R>(coord), ddc::get<Theta>(coord), a, b);
  }
};

/// The RHS of the equation for a fixed batch.
class UnbatchedRHS {
  double m_a;
  double m_b;

public:
  UnbatchedRHS(double a, double b) : m_a(a), m_b(b) {}

  KOKKOS_FUNCTION double operator()(CoordRTheta const &coord) const {
    return rho_func(ddc::get<R>(coord), ddc::get<Theta>(coord), m_a, m_b);
  }
};

} // namespace

std::unique_ptr<IPolarPoissonLikeSolver<IdxRangeRTheta, IdxRangeBatched>>
initialise_batched_solver(
    PC_tree_t const &conf_gyselalibxx,
    DiscretePoloidalCSSplineMapping<X, Y, SplineRThetaEvaluatorConstBound, R,
                                    Theta> const &discrete_mapping,
    SplineInterpolatorRThetaConst const &interpolator,
    IdxRangeBatched const idx_range_batched);
