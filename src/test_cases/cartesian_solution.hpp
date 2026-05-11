#pragma once
#include <ddc/ddc.hpp>

#include <inverse_jacobian_matrix.hpp>
#include <math_tools.hpp>
#include <coord_transformation_tools.hpp>
#include <ddc_aliases.hpp>
#include <geometry_r_theta.hpp>

/**
 * @brief Define a Cartesian solution of the Poisson-like equation.
 *
 * The solution is given by
 * * @f$ \phi (x,y) = C (1+r(x,y))^6  (1 - r(x,y))^6 \cos(2\pi x-x0) \sin(2\pi
 * y-y0) @f$,
 *
 * with @f$C = 2^{12}1e-4 @f$.
 *
 * Its x-derivative is
 *  * @f$ \partial_x \phi(x,y) = C \cdot 12 (r(x,y)^2 -1)^5 \cdot r(x,y) \cdot
 * \partial_x r(x,y)\cdot \cos(2\pi x) \sin(2\pi y)
 *  - C  (r(x,y)^2 -1)^6 \cdot 2\pi  \sin(2\pi x) \sin(2\pi y), @f$
 *
 * and its y-derivative,
 *  * @f$ \partial_y \phi(x,y) = C \cdot 12 (r(x,y)^2 -1)^5 \cdot r(x,y) \cdot
 * \partial_y r(x,y) \cdot \cos(2\pi x) \sin(2\pi y)
 *  + C  (r(x,y)^2 -1)^6 \cdot 2\pi  \cos(2\pi x) \cos(2\pi y). @f$
 */
template <class CurvilinearToCartesian>
class CartesianSolution
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
    InverseJacobianMatrix<CurvilinearToCartesian> m_inverse_jacobian;
    double m_x0;
    double m_y0;

public:
    /**
     * @brief Instantiate a CartesianSolution.
     *
     * @param[in] coordinate_converter
     *      The mapping function which converts the logical (polar)
     *      coordinates into the physical (Cartesian) coordinates.
     */
    explicit CartesianSolution(
            CurvilinearToCartesian const& coordinate_converter,
            double x0,
            double y0)
        : m_coordinate_converter(coordinate_converter)
        , m_inverse_jacobian(coordinate_converter)
        , m_x0(x0)
        , m_y0(y0)
    {
    }

	KOKKOS_DEFAULTED_FUNCTION CartesianSolution(CartesianSolution const&) = default;

    KOKKOS_FUNCTION double operator()(Coord<R, Theta> const& coord) const
    {
        const double s = ddc::get<R>(coord);
        const Coord<X, Y> cart_coord
                = m_coordinate_converter(coord);
        const double x = ddc::get<X>(cart_coord) - m_x0;
        const double y = ddc::get<Y>(cart_coord) - m_y0;
        return 1e-4 * ipow(1 + s, 6) * ipow(1 - s, 6) * std::cos(2 * M_PI * x)
               * std::sin(2 * M_PI * y) / ipow(0.5, 12);
    }
};
