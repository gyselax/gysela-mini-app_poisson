#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "core/math/Random.h"
#include "hyteg/boundary/BoundaryConditions.hpp"
#include "hyteg/mesh/MeshInfo.hpp"
#include "hyteg/mesh/micro/MicroMesh.hpp"
#include "hyteg/mesh/micro/PolarGridBuild.hpp"
#include "hyteg/mesh/micro/PolarGridIO.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/solvers/CGSolver.hpp"
#include "hyteg/solvers/ChebyshevSmoother.hpp"
#include "hyteg/solvers/GeometricMultigridSolver.hpp"

#include "ddc_alias_inline_functions.hpp"
#include "ipolar_poisson_like_solver.hpp"

// Wraps HyTeG's PolarGrids solver as an IPolarPoissonLikeSolver. The grid row
// corresponding to r=0 is replaced by r=1e-5.
template <
        class ToPhysicalMapping,
        class GridR,
        class GridTheta,
        class FunctionType,
        class OperatorType,
        class MassType,
        class RestrictionType,
        class ProlongationType,
        class MicroMeshFuncType>
class HyTegPoissonLikeSolver
    : public IPolarPoissonLikeSolver<IdxRange<GridR, GridTheta>, IdxRange<GridR, GridTheta>>
{
    using R = typename ToPhysicalMapping::curvilinear_tag_r;
    using Theta = typename ToPhysicalMapping::curvilinear_tag_theta;
    using X = typename ToPhysicalMapping::cartesian_tag_x;
    using Y = typename ToPhysicalMapping::cartesian_tag_y;

    using IdxRangeR = IdxRange<GridR>;
    using IdxRangeTheta = IdxRange<GridTheta>;
    using IdxRangeRTheta = IdxRange<GridR, GridTheta>;

    using real_t = walberla::real_t;
    using uint_t = walberla::uint_t;

    static constexpr unsigned int polynomial_degree
            = std::is_same_v<MicroMeshFuncType, hyteg::P1VectorFunction<real_t>> ? 1 : 2;

    // Determine number of levels
    static unsigned int max_num_levels(std::size_t rSize, std::size_t thetaSize)
    {
        std::size_t const numRadialIntervals = rSize - 1;

        unsigned int numLevels = 1;
        std::size_t factor = 1;
        while (true) {
            std::size_t const nextFactor = factor * 2;
            bool const rDivides = numRadialIntervals % nextFactor == 0;
            bool const thetaDivides = thetaSize % nextFactor == 0;
            bool const thetaHasEnoughPoints = thetaSize / nextFactor >= 3;
            if (!rDivides || !thetaDivides || !thetaHasEnoughPoints) {
                break;
            }
            factor = nextFactor;
            ++numLevels;
        }
        return numLevels;
    }

private:
    std::vector<real_t> rFine;
    std::vector<real_t> thetaFine;
    uint_t numLevels;
    uint_t const minLevel = 0;
    uint_t maxLevel;

    std::shared_ptr<hyteg::PrimitiveStorage> storage;
    std::shared_ptr<hyteg::micromesh::MicroMesh> microMesh;
    std::shared_ptr<MicroMeshFuncType> microMeshFunc;
    hyteg::BoundaryCondition bc;

    std::unique_ptr<FunctionType> alpha_fe;
    std::unique_ptr<FunctionType> beta_fe;
    std::unique_ptr<FunctionType> rhsNodal;
    std::unique_ptr<FunctionType> u;
    std::unique_ptr<FunctionType> f;

    std::unique_ptr<OperatorType> A;
    std::unique_ptr<MassType> M;

    int cg_max_iter;
    real_t cg_tol;

    std::shared_ptr<hyteg::CGSolver<OperatorType>> coarseGridSolver;
    std::shared_ptr<hyteg::GeometricMultigridSolver<OperatorType>> gmgPreconditioner;
    std::unique_ptr<hyteg::CGSolver<OperatorType>> pcgSolver;

public:
    HyTegPoissonLikeSolver(
            ToPhysicalMapping const& to_physical,
            IdxRangeRTheta idx_range,
            std::optional<int> num_levels_cap = std::nullopt,
            std::optional<int> cg_max_iter = std::nullopt,
            std::optional<double> cg_tol = std::nullopt)
        : cg_max_iter(cg_max_iter.value_or(100))
        , cg_tol(real_t(cg_tol.value_or(1e-9)))
    {
        IdxRangeR const idx_range_r(idx_range);
        IdxRangeTheta const idx_range_theta(idx_range);

        // Gysela keeps the true r=0 pole point, but HyTeG does not mesh through r=0.
        // The first row is therefore placed at pole_radius.
        // The coefficient/RHS values still come from Gysela's true r=0 data.
        real_t const pole_radius = real_t(1e-5);

        // Sloppy: copying elements one by one from DDC into a std::vector.
        rFine.resize(idx_range_r.size());
        {
            std::size_t i = 0;
            for (Idx<GridR> const idx : idx_range_r) {
                rFine[i++] = real_t(double(ddc::coordinate(idx)));
            }
        }
        rFine.front() = pole_radius;

        // Sloppy: copying elements one by one from DDC into a std::vector.
        thetaFine.resize(idx_range_theta.size());
        {
            std::size_t i = 0;
            for (Idx<GridTheta> const idx : idx_range_theta) {
                thetaFine[i++] = real_t(double(ddc::coordinate(idx)));
            }
        }

        unsigned int const level_cap = max_num_levels(rFine.size(), thetaFine.size());
        numLevels = level_cap;
        if (num_levels_cap.has_value()) {
            numLevels = std::min<unsigned int>(level_cap, num_levels_cap.value());
        }
        numLevels = std::max<unsigned int>(numLevels, 1);

        // A P2 MicroMesh at level L has the geometric resolution of a P1 MicroMesh at level
        // L+1 (see hyteg::buildPolarMicroMesh), so its top level is one less than the P1 case.
        maxLevel = polynomial_degree == 1 ? numLevels - 1 : numLevels - 2;

        auto const [coarseRadii, coarseThetas]
                = hyteg::MeshInfo::annulusBaseBreakpoints(rFine, thetaFine, numLevels);
        hyteg::MeshInfo const meshInfo
                = hyteg::MeshInfo::meshAnnulusBase(coarseRadii, coarseThetas);

        hyteg::SetupPrimitiveStorage setupStorage(
                meshInfo,
                walberla::uint_c(walberla::mpi::MPIManager::instance()->numProcesses()));
        storage = std::make_shared<hyteg::PrimitiveStorage>(setupStorage);

        bc.createNeumannBC("innerRing", 1);
        bc.createDirichletBC("outerRing", 2);

        // Geometry mapping
        hyteg::PolarGridMapT const map = [&to_physical](real_t r, real_t theta) {
            Coord<R, Theta> const rtheta {double(r), double(theta)};
            Coord<X, Y> const xy = to_physical(rtheta);
            return hyteg::Point3D(double(Coord<X>(xy)), double(Coord<Y>(xy)), real_t(0));
        };

        microMesh = std::make_shared<
                hyteg::micromesh::MicroMesh>(storage, minLevel, maxLevel, polynomial_degree, 2);
        hyteg::buildPolarMicroMesh(*microMesh, rFine, thetaFine, numLevels, map);
        storage->setMicroMesh(microMesh);
        microMeshFunc = get_micromesh_func();

        alpha_fe = std::make_unique<FunctionType>("alpha", storage, minLevel, maxLevel, bc);
        beta_fe = std::make_unique<FunctionType>("beta", storage, minLevel, maxLevel, bc);
        rhsNodal = std::make_unique<FunctionType>("rhsNodal", storage, minLevel, maxLevel, bc);
        u = std::make_unique<FunctionType>("u", storage, minLevel, maxLevel, bc);
        f = std::make_unique<FunctionType>("f", storage, minLevel, maxLevel, bc);

        // Homogeneous Dirichlet BC on the outer boundary: u=0 there
        for (uint_t level = minLevel; level <= maxLevel; level++) {
            u->interpolate(
                    [](hyteg::Point3D const&) { return real_t(0); },
                    level,
                    hyteg::DirichletBoundary);
        }

        // Operators and the solver hierarchy are built in update_coefficients(), once
        // alpha/beta are known.
    }

    void update_coefficients(DConstField<IdxRangeRTheta> alpha, DConstField<IdxRangeRTheta> beta)
            override
    {
        std::vector<real_t> const alphaTensorGrid = extract_tensor_grid(alpha);
        std::vector<real_t> const betaTensorGrid = extract_tensor_grid(beta);

        // injectDown = true: GMG applies A at every level from minLevel to maxLevel, so
        // alpha/beta must be populated on all of them
        hyteg::readPolarTensorGrid(*alpha_fe, rFine, thetaFine, numLevels, alphaTensorGrid, true);
        hyteg::readPolarTensorGrid(*beta_fe, rFine, thetaFine, numLevels, betaTensorGrid, true);

        A = std::make_unique<
                OperatorType>(storage, minLevel, maxLevel, *alpha_fe, *beta_fe, *microMeshFunc);
        M = std::make_unique<MassType>(storage, minLevel, maxLevel, *microMeshFunc);
        A->computeInverseDiagonalOperatorValues();

        auto smoother = std::make_shared<
                hyteg::ChebyshevSmoother<OperatorType>>(storage, minLevel, maxLevel);
        {
            // Spectral radii estimates for Chebyshev smoother
            std::vector<real_t> spectralRadii(maxLevel - minLevel + 1);
            FunctionType eigenVec("eigenVec", storage, minLevel, maxLevel);
            FunctionType eigenTmp("eigenTmp", storage, minLevel, maxLevel);
            for (uint_t level = minLevel; level <= maxLevel; level++) {
                eigenVec.interpolate(
                        [](hyteg::Point3D const&) { return walberla::math::realRandom(); },
                        level);
                spectralRadii[level - minLevel] = hyteg::chebyshev::
                        estimateRadius(*A, level, 20, storage, eigenVec, eigenTmp);
            }
            smoother->setupCoefficients(3, spectralRadii);
        }

        auto restriction = std::make_shared<RestrictionType>();
        auto prolongation = std::make_shared<ProlongationType>();

        // Coarse grid tol, chosen to be 1e-2 * fine grid tol
        real_t const coarseTolerance = cg_tol * real_t(1e-2);
        coarseGridSolver = std::make_shared<hyteg::CGSolver<
                OperatorType>>(storage, minLevel, minLevel, 1000, coarseTolerance, real_t(0));
        coarseGridSolver->setPrintInfo(false);

        gmgPreconditioner = std::make_shared<hyteg::GeometricMultigridSolver<OperatorType>>(
                storage,
                smoother,
                coarseGridSolver,
                restriction,
                prolongation,
                minLevel,
                maxLevel,
                2,
                2);

        pcgSolver = std::make_unique<hyteg::CGSolver<OperatorType>>(
                storage,
                minLevel,
                maxLevel,
                cg_max_iter,
                cg_tol,
                real_t(0),
                gmgPreconditioner);
        pcgSolver->setPrintInfo(false);
    }

    void operator()(DField<IdxRangeRTheta> phi, DConstField<IdxRangeRTheta> rho) const override
    {
        std::vector<real_t> const rhoTensorGrid = extract_tensor_grid(rho);

        hyteg::readPolarTensorGrid(*rhsNodal, rFine, thetaFine, numLevels, rhoTensorGrid, false);
        M->apply(*rhsNodal, *f, maxLevel, hyteg::All);

        pcgSolver->solve(*A, *u, *f, maxLevel);

        std::vector<real_t> solutionTensorGrid;
        hyteg::writePolarTensorGrid(*u, rFine, thetaFine, numLevels, solutionTensorGrid);

        write_tensor_grid_to_phi(solutionTensorGrid, phi);
    }

private:
    std::shared_ptr<MicroMeshFuncType> get_micromesh_func() const
    {
        if constexpr (polynomial_degree == 1) {
            return microMesh->p1Mesh();
        } else {
            return microMesh->p2Mesh();
        }
    }

    // Copies a gyselalibxx (r,theta) field to host and flattens it into HyTeG's
    // rIndex*thetaFine.size()+thetaIndex tensor-grid layout.
    std::vector<real_t> extract_tensor_grid(DConstField<IdxRangeRTheta> field) const
    {
        auto const host_alloc = ddc::create_mirror_view_and_copy(Kokkos::HostSpace(), field);
        auto const view_2d = get_const_field(host_alloc).allocation_kokkos_view();

        // This is still a real copy: hyteg::readPolarTensorGrid only accepts a
        // std::vector<real_t>, not a Kokkos::View, so view_2d's memory can't be handed
        // through directly.
        std::vector<real_t> tensorGrid(view_2d.size());
        std::transform(
                view_2d.data(),
                view_2d.data() + view_2d.size(),
                tensorGrid.begin(),
                [](double v) { return real_t(v); });
        return tensorGrid;
    }

    // Inverse of extract_tensor_grid. Gysela's true r=0 is a single physical point,
    // so the pole row is overwritten with its theta-average.
    void write_tensor_grid_to_phi(std::vector<real_t> const& tensorGrid, DField<IdxRangeRTheta> phi)
            const
    {
        auto phi_alloc_host = ddc::create_mirror_view(Kokkos::HostSpace(), phi);
        auto const view_2d = get_field(phi_alloc_host).allocation_kokkos_view();

        // Still a real copy: hyteg::writePolarTensorGrid only produces a std::vector<real_t>,
        // not a Kokkos::View, so it can't be written directly into view_2d's memory.
        std::transform(tensorGrid.begin(), tensorGrid.end(), view_2d.data(), [](real_t v) {
            return double(v);
        });

        double pole_sum = 0.0;
        for (std::size_t thetaIndex = 0; thetaIndex < thetaFine.size(); ++thetaIndex) {
            pole_sum += double(tensorGrid[thetaIndex]);
        }
        double const pole_value = pole_sum / double(thetaFine.size());
        for (std::size_t thetaIndex = 0; thetaIndex < thetaFine.size(); ++thetaIndex) {
            view_2d.data()[thetaIndex] = pole_value;
        }

        ddc::parallel_deepcopy(phi, get_field(phi_alloc_host));
    }
};
