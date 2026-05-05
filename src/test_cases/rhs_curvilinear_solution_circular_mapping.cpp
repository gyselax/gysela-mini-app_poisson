// SPDX-License-Identifier: MIT
#include <circular_to_cartesian.hpp>
#include <math_tools.hpp>

#include "curvilinear_solution.hpp"
#include "rhs.hpp"

template <>
KOKKOS_FUNCTION double ManufacturedRHS<CurvilinearSolution<CircularToCartesian<R, Theta, X, Y>>>::
        solution_at_pole(Coord<R, Theta> const& coord) const
{
    return 0.0;
}

template <>
KOKKOS_FUNCTION double ManufacturedRHS<CurvilinearSolution<CircularToCartesian<R, Theta, X, Y>>>::
        non_singular_solution(Coord<R, Theta> const& coord) const
{
    const double r = ddc::get<R>(coord);
    const double theta = ddc::get<Theta>(coord);

    const double cos_11_theta = Kokkos::cos(11.0 * theta);

    const double tanh_term = Kokkos::tanh(20.0 * r - 14.0);
    const double coeff_alpha = Kokkos::exp(tanh_term);

    const double minus_pow_6 = ipow(r - 1.0, 6);

    return 0.4096 * ipow(r, 6) * minus_pow_6 * coeff_alpha * cos_11_theta
           - ipow(r, 4)
                     * (r
                                * (12.288 * r * ipow(r - 1.0, 4) * cos_11_theta
                                   + 17.2032 * ipow(r - 1.0, 5) * cos_11_theta)
                                * Kokkos::exp(-tanh_term)
                        + r
                                  * (2.4576 * r * ipow(r - 1.0, 5) * cos_11_theta
                                     + 2.4576 * minus_pow_6 * cos_11_theta)
                                  * (20.0 * ipow(tanh_term, 2) - 20.0) * Kokkos::exp(-tanh_term)
                        - 49.5616 * minus_pow_6 * Kokkos::exp(-tanh_term) * cos_11_theta
                        + 6.0
                                  * (2.4576 * r * ipow(r - 1.0, 5) * cos_11_theta
                                     + 2.4576 * minus_pow_6 * cos_11_theta)
                                  * Kokkos::exp(-tanh_term));
}

template class ManufacturedRHS<CurvilinearSolution<CircularToCartesian<R, Theta, X, Y>>>;
