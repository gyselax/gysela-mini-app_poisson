#pragma once
#include <ddc/ddc.hpp>

#include <coord_transformation_tools.hpp>
#include <ddc_aliases.hpp>
#include <geometry_r_theta.hpp>
#include <math_tools.hpp>

/**
 * @brief Define a curvilinear solution of the Poisson-like equation.
 *
 * The solution is given by
 * * @f$ \phi(x, y) = C r(x,y)^6 (r(x,y) -1)^6 \cos(m\theta) @f$,
 *
 * with @f$C = 2^{12}1e-4 @f$ and @f$ m = 11 @f$.
 */
template <class CurvilinearToCartesian>
class CurvilinearSolution
{
public:
    /**
     * @brief Type the mapping function which converts the logical (polar)
     * coordinates into the physical (Cartesian) coordinates.
     */
    using coordinate_converter_type = CurvilinearToCartesian;
    static_assert(is_coord_transform_with_o_point_v<CurvilinearToCartesian>);

private:
    /**
     * @brief The mapping function which converts the logical (polar)
     * coordinates into the physical (Cartesian) coordinates.
     */
    CurvilinearToCartesian const& m_coordinate_converter;

public:
    /**
     * @brief Instantiate a CurvilinearSolution.
     *
     * @param[in] coordinate_converter
     *      The mapping function which converts the logical (polar)
     *      coordinates into the physical (Cartesian) coordinates.
     */
    explicit CurvilinearSolution(
            CurvilinearToCartesian const& coordinate_converter,
            double x0,
            double y0)
        : m_coordinate_converter(coordinate_converter)
    {
    }

	KOKKOS_DEFAULTED_FUNCTION CurvilinearSolution(CurvilinearSolution const&) = default;

    KOKKOS_FUNCTION double operator()(Coord<R, Theta> const& coord) const
    {
        const double s = ddc::get<R>(coord);
        const double t = ddc::get<Theta>(coord);
        return 1e-4 * ipow(s, 6) * ipow(s - 1, 6) / ipow(0.5, 12) * std::cos(11 * t);
    }
};
