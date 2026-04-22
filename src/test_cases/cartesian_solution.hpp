#pragma once
#include <ddc/ddc.hpp>
#include <inverse_jacobian_matrix.hpp>
#include <math_tools.hpp>

#include "i_solution.hpp"

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
class CartesianSolution : public ISolution<CurvilinearToCartesian> {
   private:
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
        CurvilinearToCartesian const& coordinate_converter, double x0,
        double y0)
        : ISolution<CurvilinearToCartesian>(coordinate_converter),
          m_inverse_jacobian(coordinate_converter),
          m_x0(x0),
          m_y0(y0) {}

    double operator()(Coord<R, Theta> const& coord) const final {
        const double s = ddc::get<R>(coord);
        const Coord<X, Y> cart_coord =
            ISolution<CurvilinearToCartesian>::m_coordinate_converter(coord);
        const double x = ddc::get<X>(cart_coord) - m_x0;
        const double y = ddc::get<Y>(cart_coord) - m_y0;
        return 1e-4 * ipow(1 + s, 6) * ipow(1 - s, 6) * std::cos(2 * M_PI * x) *
               std::sin(2 * M_PI * y) / ipow(0.5, 12);
    }
};
