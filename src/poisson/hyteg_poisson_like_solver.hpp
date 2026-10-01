// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <ddc/ddc.hpp>

#include "core/logging/Logging.h"
#include "core/math/Random.h"
#include "core/mpi/MPIManager.h"
#include "hyteg/gridtransferoperators/P1toP1LinearProlongation.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearRestriction.hpp"
#include "hyteg/mesh/MeshInfo.hpp"
#include "hyteg/mesh/micro/MicroMesh.hpp"
#include "hyteg/mesh/micro/PolarMicroMesh.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/solvers/CGSolver.hpp"
#include "hyteg/solvers/ChebyshevSmoother.hpp"
#include "hyteg/solvers/GeometricMultigridSolver.hpp"
#include "hyteg_operators/operators/div_alpha_grad_plus_beta_mass/P1ElementwiseDivAlphaGradPlusBetaMassP1CoeffsParametricP1Map.hpp"
#include "hyteg_operators/operators/mass/P1ElementwiseMassParametricP1Map.hpp"

#include "ddc_alias_inline_functions.hpp"
#include "ipolar_poisson_like_solver.hpp"

/**
 * @brief Solves -div(alpha grad phi) + beta phi = rho, phi = 0 on the outer boundary, with HyTeG.
 *
 * Stores the solution at mesh nodes and uses linear (P1) elements for the solution and geometry.
 * CG solves the resulting system, using a geometric multigrid V-cycle as its preconditioner.
 * Call update_coefficients() before the first solve.
 *
 * Input values are copied from the (r, theta) grid to matching mesh nodes. On output, values are copied
 * back to the grid; points without a mesh node are filled by interpolation from nearby nodes.
 * The centre has one stored value, copied to every angle in the r = 0 output row.
 * Every MPI rank must supply the complete tensor grid.
 *
 * The radial grid must start at r = 0. Angles must lie in [0, 2 pi), without a duplicate endpoint.
 * By default, use as many multigrid levels as the grid allows. For L levels, (number of r points - 1) and
 * the number of theta points must be divisible by 2^(L - 1), with at least three coarse angles.
 *
 * @tparam ToPhysicalMapping The mapping from (r, theta) to (x, y).
 * @tparam GridR The radial grid.
 * @tparam GridTheta The poloidal grid.
 */
template <class ToPhysicalMapping, class GridR, class GridTheta>
class HyTegPoissonLikeSolver
    : public IPolarPoissonLikeSolver<IdxRange<GridR, GridTheta>, IdxRange<GridR, GridTheta>>
{
    using base_type
            = IPolarPoissonLikeSolver<IdxRange<GridR, GridTheta>, IdxRange<GridR, GridTheta>>;
    using typename base_type::const_field_type;
    using typename base_type::field_type;

    using X = typename ToPhysicalMapping::cartesian_tag_x;
    using Y = typename ToPhysicalMapping::cartesian_tag_y;

    using IdxRangeR = IdxRange<GridR>;
    using IdxRangeTheta = IdxRange<GridTheta>;
    using IdxRangeRTheta = IdxRange<GridR, GridTheta>;
    using IdxRTheta = Idx<GridR, GridTheta>;

    using real_t = walberla::real_t;
    using uint_t = walberla::uint_t;

    // Values at mesh nodes, stored separately for each multigrid level.
    using Function = hyteg::P1Function<real_t>;
    using Operator = hyteg::operatorgeneration::
            P1ElementwiseDivAlphaGradPlusBetaMassP1CoeffsParametricP1Map;
    using Mass = hyteg::operatorgeneration::P1ElementwiseMassParametricP1Map;

    static_assert(std::is_same_v<real_t, double>, "HyTeG must be built with double precision");
    static_assert(
            std::is_same_v<typename base_type::layout_space, Kokkos::layout_right>,
            "The tensor-grid layout needs theta to vary fastest");

    static constexpr uint_t s_min_level = 0;
    // A positive absolute tolerance lets CG stop when the initial residual is exactly zero.
    static constexpr real_t s_cg_absolute_tolerance = 1e-14;

private:
    // Input radii and angles. The x and y arrays store the physical position of each grid point,
    // at index i * m_theta.size() + j for the point (m_r[i], m_theta[j]).
    std::vector<real_t> m_r;
    std::vector<real_t> m_theta;
    std::vector<real_t> m_x;
    std::vector<real_t> m_y;

    uint_t m_max_level;

    std::shared_ptr<hyteg::PrimitiveStorage> m_storage;
    std::shared_ptr<hyteg::micromesh::MicroMesh> m_micro_mesh;

    std::unique_ptr<Function> m_alpha;
    std::unique_ptr<Function> m_beta;
    std::unique_ptr<Function> m_rho;
    std::unique_ptr<Function> m_load;
    std::unique_ptr<Function> m_phi;

    std::unique_ptr<Mass> m_mass;
    std::unique_ptr<Operator> m_operator;

    uint_t m_cg_max_iter;
    real_t m_cg_tol;
    std::shared_ptr<hyteg::GeometricMultigridSolver<Operator>> m_multigrid;
    std::unique_ptr<hyteg::CGSolver<Operator>> m_cg;

public:
    /**
     * @brief Build the mesh and allocate values at its nodes on every multigrid level.
     * @param[in] to_physical The mapping from (r, theta) to (x, y).
     * @param[in] idx_range The (r, theta) grid of the fields passed to the solver.
     * @param[in] num_levels Requested number of levels, capped by the grid size. Defaults to all available levels.
     * @param[in] cg_max_iter The maximum number of CG iterations (default 100).
     * @param[in] cg_tol Residual tolerance relative to the initial residual (default 1e-9).
     *                   CG also stops when the residual norm is below 1e-14.
     */
    HyTegPoissonLikeSolver(
            ToPhysicalMapping const& to_physical,
            IdxRangeRTheta idx_range,
            std::optional<int> num_levels = std::nullopt,
            std::optional<int> cg_max_iter = std::nullopt,
            std::optional<double> cg_tol = std::nullopt)
        : m_cg_max_iter(cg_max_iter.value_or(100))
        , m_cg_tol(cg_tol.value_or(1e-9))
    {
        for (Idx<GridR> const ir : IdxRangeR(idx_range)) {
            m_r.push_back(double(ddc::coordinate(ir)));
        }
        for (Idx<GridTheta> const itheta : IdxRangeTheta(idx_range)) {
            m_theta.push_back(double(ddc::coordinate(itheta)));
        }
        if (m_r.size() < 2 || m_theta.size() < 3) {
            throw std::runtime_error("The HyTeg grid needs at least two radii and three angles");
        }
        if (m_r.front() != 0.0) {
            throw std::runtime_error(
                    "The HyTeg solver needs r_min = 0: its disc mesh has a node at the centre");
        }

        // Choose how many times the grid can be coarsened. Level 0 is the coarsest level.
        uint_t const available_levels = num_available_levels(m_r.size(), m_theta.size());
        uint_t const levels = num_levels.has_value()
                                      ? std::clamp<uint_t>(num_levels.value(), 1, available_levels)
                                      : available_levels;
        if (levels < 4) {
            WALBERLA_LOG_WARNING_ON_ROOT(
                    "HyTeg Poisson solver: only "
                    << levels << " multigrid levels for " << m_r.size() << " x " << m_theta.size()
                    << " (r x theta) grid points.");
        }
        m_max_level = levels - 1;

        // Build the coarse mesh from every 2^m_max_level-th radius and angle.
        // Refinement adds nodes at points of the input tensor grid, but does not use every grid point.
        auto const [r_coarse, theta_coarse]
                = hyteg::micromesh::polarCoarseLists(m_r, m_theta, m_max_level);
        std::vector<real_t> transition_radii;
        hyteg::SetupPrimitiveStorage setup_storage(
                hyteg::MeshInfo::meshPolar(r_coarse, theta_coarse, transition_radii),
                walberla::uint_c(walberla::mpi::MPIManager::instance()->numProcesses()));
        setup_storage.setMeshBoundaryFlagsOnBoundary(1, 0, true);
        m_storage = std::make_shared<hyteg::PrimitiveStorage>(setup_storage);

        WALBERLA_LOG_INFO_ON_ROOT(
                "HyTeg Poisson solver: " << m_r.size() << " x " << m_theta.size()
                                         << " (r x theta) grid points, " << levels
                                         << " multigrid levels");

        m_micro_mesh = std::make_shared<hyteg::micromesh::MicroMesh>(
                m_storage,
                s_min_level,
                m_max_level,
                /*polynomialDegree*/ 1,
                /*dimension*/ 2);
        hyteg::micromesh::polarMicroMesh(*m_micro_mesh, m_r, m_theta);
        m_storage->setMicroMesh(m_micro_mesh);

        // Map each (r, theta) grid point to (x, y), then set the mesh-node positions on every level.
        DFieldMem<IdxRangeRTheta> x_alloc(idx_range);
        DFieldMem<IdxRangeRTheta> y_alloc(idx_range);
        DField<IdxRangeRTheta> x = get_field(x_alloc);
        DField<IdxRangeRTheta> y = get_field(y_alloc);
        ddc::parallel_for_each(
                Kokkos::DefaultExecutionSpace(),
                idx_range,
                KOKKOS_LAMBDA(IdxRTheta const irtheta) {
                    Coord<X, Y> const position = to_physical(ddc::coordinate(irtheta));
                    x(irtheta) = ddc::get<X>(position);
                    y(irtheta) = ddc::get<Y>(position);
                });
        m_x = host_values(get_const_field(x));
        m_y = host_values(get_const_field(y));
        hyteg::micromesh::readPolarTensorGrid(
                m_micro_mesh->p1Mesh()->component(0),
                m_r,
                m_theta,
                m_max_level,
                m_x);
        hyteg::micromesh::readPolarTensorGrid(
                m_micro_mesh->p1Mesh()->component(1),
                m_r,
                m_theta,
                m_max_level,
                m_y);

        m_alpha = std::make_unique<Function>("alpha", m_storage, s_min_level, m_max_level);
        m_beta = std::make_unique<Function>("beta", m_storage, s_min_level, m_max_level);
        m_rho = std::make_unique<Function>("rho", m_storage, s_min_level, m_max_level);
        m_load = std::make_unique<Function>("load", m_storage, s_min_level, m_max_level);
        m_phi = std::make_unique<Function>("phi", m_storage, s_min_level, m_max_level);

        // Start the first solve from zero. Later solves reuse phi; its outer boundary stays zero.
        for (uint_t level = s_min_level; level <= m_max_level; ++level) {
            m_phi->interpolate(real_t(0), level, hyteg::All);
        }

        m_mass = std::make_unique<
                Mass>(m_storage, s_min_level, m_max_level, *m_micro_mesh->p1Mesh());

        // Build the operator and solvers later, when update_coefficients() supplies alpha and beta.
    }

    /**
     * @brief Update the coefficients alpha and beta, and rebuild the operator and the multigrid solver.
     * @param[in] alpha The values of alpha at the grid points.
     * @param[in] beta The values of beta at the grid points.
     */
    void update_coefficients(const_field_type alpha, const_field_type beta) override
    {
        // Copy alpha and beta to every level because multigrid solves on coarse grids too.
        read_field(*m_alpha, alpha);
        read_field(*m_beta, beta);

        m_operator = std::make_unique<Operator>(
                m_storage,
                s_min_level,
                m_max_level,
                *m_alpha,
                *m_beta,
                *m_micro_mesh->p1Mesh());
        m_operator->computeInverseDiagonalOperatorValues();

        // Set up the Chebyshev smoother using a separate spectral-radius estimate for each level.
        auto smoother = std::make_shared<
                hyteg::ChebyshevSmoother<Operator>>(m_storage, s_min_level, m_max_level);
        {
            std::vector<real_t> spectral_radii;
            Function power_vector("power_vector", m_storage, s_min_level, m_max_level);
            Function power_tmp("power_tmp", m_storage, s_min_level, m_max_level);
            for (uint_t level = s_min_level; level <= m_max_level; ++level) {
                power_vector.interpolate(
                        [](hyteg::Point3D const&) { return walberla::math::realRandom(); },
                        level,
                        hyteg::Inner);
                power_vector.interpolate(real_t(0), level, hyteg::DirichletBoundary);
                spectral_radii.push_back(hyteg::chebyshev::estimateRadius(
                        *m_operator,
                        level,
                        20,
                        m_storage,
                        power_vector,
                        power_tmp));
            }
            smoother->setupCoefficients(3, spectral_radii);
        }

        // Use CG to approximately solve the correction equation on the coarsest grid.
        auto coarse_solver = std::make_shared<hyteg::CGSolver<Operator>>(
                m_storage,
                s_min_level,
                s_min_level,
                1000,
                real_t(1e-7),
                s_cg_absolute_tolerance * real_t(1e-2));

        m_multigrid = std::make_shared<hyteg::GeometricMultigridSolver<Operator>>(
                m_storage,
                smoother,
                coarse_solver,
                std::make_shared<hyteg::P1toP1LinearRestriction<real_t>>(),
                std::make_shared<hyteg::P1toP1LinearProlongation<real_t>>(),
                s_min_level,
                m_max_level,
                2,
                2);

        m_cg = std::make_unique<hyteg::CGSolver<Operator>>(
                m_storage,
                s_min_level,
                m_max_level,
                m_cg_max_iter,
                m_cg_tol,
                s_cg_absolute_tolerance,
                m_multigrid);
        m_cg->setPrintInfo(true);
    }

    /**
     * @brief Solve the equation for phi.
     * @param[out] phi The solution at the grid points.
     * @param[in] rho The right-hand side at the grid points.
     */
    void operator()(field_type phi, const_field_type rho) const override
    {
        if (!m_cg) {
            throw std::logic_error("Call update_coefficients() before solving with HyTeg");
        }

        // Copy rho to mesh nodes. Multiply by the mass matrix to form the load vector: load = M * rho.
        read_field(*m_rho, rho);
        m_mass->apply(*m_rho, *m_load, m_max_level, hyteg::All);

        m_cg->solve(*m_operator, *m_phi, *m_load, m_max_level);

        // Write the solution to a CPU buffer, filling missing grid points by interpolation.
        // Then copy the result to phi in its original memory space.
        auto phi_host_alloc = ddc::create_mirror_view(Kokkos::HostSpace(), phi);
        auto const phi_host = get_field(phi_host_alloc).allocation_kokkos_view();
        WALBERLA_CHECK(phi_host.span_is_contiguous());
        hyteg::micromesh::writePolarTensorGrid(
                *m_phi,
                m_r,
                m_theta,
                m_max_level,
                m_x,
                m_y,
                std::span<real_t>(phi_host.data(), phi_host.size()));
        ddc::parallel_deepcopy(phi, get_field(phi_host_alloc));
    }

private:
    // Count levels by halving the radial intervals and angles while both counts remain divisible by two.
    // Keep at least three angles on the coarsest grid.
    static uint_t num_available_levels(std::size_t num_r, std::size_t num_theta)
    {
        uint_t levels = 1;
        std::size_t stride = 2;
        while ((num_r - 1) % stride == 0 && num_theta % stride == 0 && num_theta / stride >= 3) {
            ++levels;
            stride *= 2;
        }
        return levels;
    }

    // Copy field's grid values into target's mesh-node values on every level.
    void read_field(Function& target, const_field_type field) const
    {
        // The host mirror may share the original memory.
        // Wait for pending Kokkos operations before reading it on the CPU.
        Kokkos::fence("HyTeg input ready");
        auto const host_alloc = ddc::create_mirror_view_and_copy(Kokkos::HostSpace(), field);
        auto const view = get_const_field(host_alloc).allocation_kokkos_view();
        WALBERLA_CHECK(view.span_is_contiguous());
        hyteg::micromesh::readPolarTensorGrid(
                target,
                m_r,
                m_theta,
                m_max_level,
                std::span<real_t const>(view.data(), view.size()));
    }

    // Return a separate CPU copy of the coordinates, with theta varying fastest.
    static std::vector<real_t> host_values(const_field_type field)
    {
        Kokkos::fence("HyTeg coordinates ready");
        auto const host_alloc = ddc::create_mirror_view_and_copy(Kokkos::HostSpace(), field);
        auto const view = get_const_field(host_alloc).allocation_kokkos_view();
        WALBERLA_CHECK(view.span_is_contiguous());
        return std::vector<real_t>(view.data(), view.data() + view.size());
    }
};
