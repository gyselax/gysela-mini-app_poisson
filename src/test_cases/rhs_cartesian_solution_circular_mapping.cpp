// SPDX-License-Identifier: MIT
#include <circular_to_cartesian.hpp>
#include <math_tools.hpp>

#include "cartesian_solution.hpp"
#include "rhs.hpp"

template <>
KOKKOS_FUNCTION double
ManufacturedRHS<CartesianSolution<CircularToCartesian<R, Theta, X, Y>>>::
    solution_at_pole(Coord<R, Theta> const& coord) const {
    return 0.0;
}

template <>
KOKKOS_FUNCTION double
ManufacturedRHS<CartesianSolution<CircularToCartesian<R, Theta, X, Y>>>::
    non_singular_solution(Coord<R, Theta> const& coord) const {
    const double r = ddc::get<R>(coord);
    const double theta = ddc::get<Theta>(coord);

    const double sin_theta = Kokkos::sin(theta);
    const double cos_theta = Kokkos::cos(theta);

    const double tanh_term = Kokkos::tanh(20.0 * r - 14.0);
    const double coeff_alpha = Kokkos::exp(tanh_term);

    const double plus_pow_6 = ipow(r + 1.0, 6);
    const double minus_pow_6 = ipow(1.0 - r, 6);

    const double cos_sin = Kokkos::cos(2.0 * M_PI * r * sin_theta);
    const double cos_cos = Kokkos::cos(2.0 * M_PI * r * cos_theta);
    const double sin_sin = Kokkos::sin(2.0 * M_PI * r * sin_theta);
    const double sin_cos = Kokkos::sin(2.0 * M_PI * r * cos_theta);

    return 0.4096 * minus_pow_6 * plus_pow_6 * coeff_alpha * sin_sin * cos_cos -
           (r * (20.0 * ipow(tanh_term, 2) - 20.0) *
                (0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta *
                     cos_sin * cos_cos -
                 0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_sin * sin_cos *
                     cos_theta +
                 2.4576 * minus_pow_6 * ipow((r + 1.0), 5) * sin_sin * cos_cos -
                 2.4576 * ipow((1.0 - r), 5) * plus_pow_6 * sin_sin * cos_cos) *
                Kokkos::exp(-tanh_term) +
            r *
                ((-1.6384) * (M_PI * M_PI) * minus_pow_6 * plus_pow_6 *
                     ipow(sin_theta, 2) * sin_sin * cos_cos -
                 3.2768 * (M_PI * M_PI) * minus_pow_6 * plus_pow_6 * sin_theta *
                     sin_cos * cos_theta * cos_sin -
                 1.6384 * (M_PI * M_PI) * minus_pow_6 * plus_pow_6 * sin_sin *
                     ipow(cos_theta, 2) * cos_cos +
                 9.8304 * M_PI * minus_pow_6 * ipow(r + 1.0, 5) * sin_theta *
                     cos_sin * cos_cos -
                 9.8304 * M_PI * minus_pow_6 * ipow(r + 1.0, 5) * sin_sin *
                     sin_cos * cos_theta +
                 12.288 * minus_pow_6 * ipow(r + 1.0, 4) * sin_sin * cos_cos -
                 9.8304 * M_PI * ipow(1.0 - r, 5) * plus_pow_6 * sin_theta *
                     cos_sin * cos_cos +
                 9.8304 * M_PI * ipow(1.0 - r, 5) * plus_pow_6 * sin_sin *
                     sin_cos * cos_theta -
                 29.4912 * ipow(1.0 - r, 5) * ipow(r + 1.0, 5) * sin_sin *
                     cos_cos +
                 12.288 * ipow(1.0 - r, 4) * plus_pow_6 * sin_sin * cos_cos) *
                Kokkos::exp(-tanh_term) +
            (0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta * cos_sin *
                 cos_cos -
             0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_sin * sin_cos *
                 cos_theta +
             2.4576 * minus_pow_6 * ipow(r + 1.0, 5) * sin_sin * cos_cos -
             2.4576 * ipow(1.0 - r, 5) * plus_pow_6 * sin_sin * cos_cos) *
                Kokkos::exp(-tanh_term) +
            ((-1.6384) * (M_PI * M_PI) * r * minus_pow_6 * plus_pow_6 *
                 ipow(sin_theta, 2) * sin_sin * cos_cos +
             3.2768 * (M_PI * M_PI) * r * minus_pow_6 * plus_pow_6 * sin_theta *
                 sin_cos * cos_theta * cos_sin -
             1.6384 * (M_PI * M_PI) * r * minus_pow_6 * plus_pow_6 * sin_sin *
                 ipow(cos_theta, 2) * cos_cos -
             0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta * cos_sin *
                 cos_cos +
             0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_sin * sin_cos *
                 cos_theta) *
                Kokkos::exp(-tanh_term)) /
               r;
}

template class ManufacturedRHS<
    CartesianSolution<CircularToCartesian<R, Theta, X, Y>>>;
