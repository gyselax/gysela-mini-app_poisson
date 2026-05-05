// SPDX-License-Identifier: MIT
#include <czarny_to_cartesian.hpp>
#include <math_tools.hpp>

#include "cartesian_solution.hpp"
#include "rhs.hpp"

template <>
KOKKOS_FUNCTION double ManufacturedRHS<CartesianSolution<CzarnyToCartesian<R, Theta, X, Y>>>::
        solution_at_pole(Coord<R, Theta> const& coord) const
{
    return 0.0;
}

template <>
KOKKOS_FUNCTION double ManufacturedRHS<CartesianSolution<CzarnyToCartesian<R, Theta, X, Y>>>::
        non_singular_solution(Coord<R, Theta> const& coord) const
{
    const double r = ddc::get<R>(coord);
    const double theta = ddc::get<Theta>(coord);
    const double epsilon = m_coordinate_converter.epsilon();
    const double e = m_coordinate_converter.e();

    const double sin_theta = Kokkos::sin(theta);
    const double cos_theta = Kokkos::cos(theta);

    const double tanh_term = Kokkos::tanh(20.0 * r - 14.0);
    const double coeff_alpha = Kokkos::exp(tanh_term);

    const double plus_pow_6 = ipow(r + 1.0, 6);
    const double minus_pow_6 = ipow(1.0 - r, 6);

    const double xi = 1. / Kokkos::sqrt(1. - epsilon * epsilon * 0.25);
    const double tmp1 = Kokkos::sqrt(epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0);

    return 0.4096 * minus_pow_6 * plus_pow_6 * coeff_alpha
                   * Kokkos::sin(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
           - ((-r)
                      * (((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                          + e * cos_theta * xi / ((2.0 - tmp1)))
                                 * (e * epsilon * r * sin_theta * cos_theta * xi
                                            / (ipow(2.0 - tmp1, 2) * tmp1)
                                    + e * sin_theta * xi / ((2.0 - tmp1)))
                         - sin_theta * cos_theta
                                   / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                      * (0.4096 * minus_pow_6 * plus_pow_6
                                 * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                            / (ipow(2.0 - tmp1, 2) * tmp1)
                                    + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                 * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                 * Kokkos::cos(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                         - 0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta
                                   * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::sin(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                   / tmp1)
                      * (20.0 * ipow(tanh_term, 2) - 20.0) * Kokkos::exp(-tanh_term)
                      / Kokkos::sqrt(
                              (-ipow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                               / (ipow(2.0 - tmp1, 2) * tmp1)
                                       + e * cos_theta * xi / ((2.0 - tmp1)))
                                              * (e * epsilon * r * sin_theta * cos_theta * xi
                                                         / (ipow(2.0 - tmp1, 2) * tmp1)
                                                 + e * sin_theta * xi / ((2.0 - tmp1)))
                                      - sin_theta * cos_theta
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                   + 1.0)),
                                     2))
                              + (ipow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                               / (ipow((2.0 - tmp1), 2) * tmp1)
                                       + e * cos_theta * xi / ((2.0 - tmp1))),
                                      2)
                                 + ipow(sin_theta, 2)
                                           / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                        * (ipow((e * epsilon * r * sin_theta * cos_theta * xi
                                                         / (ipow((2.0 - tmp1), 2) * tmp1)
                                                 + e * sin_theta * xi / ((2.0 - tmp1))),
                                                2)
                                           + ipow(cos_theta, 2)
                                                     / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0)))
              - r
                        * (((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                    / (ipow((2.0 - tmp1), 2) * tmp1)
                            + e * cos_theta * xi / ((2.0 - tmp1)))
                                   * (e * epsilon * r * sin_theta * cos_theta * xi
                                              / (ipow((2.0 - tmp1), 2) * tmp1)
                                      + e * sin_theta * xi / ((2.0 - tmp1)))
                           - sin_theta * cos_theta
                                     / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (0.4096 * minus_pow_6 * plus_pow_6
                                   * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                              / (ipow((2.0 - tmp1), 2) * tmp1)
                                      + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::cos(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1)
                        * (1.0 / 2.0
                                   * (((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                               / (ipow((2.0 - tmp1), 2) * tmp1)
                                       + e * cos_theta * xi / ((2.0 - tmp1)))
                                              * (e * epsilon * r * sin_theta * cos_theta * xi
                                                         / (ipow((2.0 - tmp1), 2) * tmp1)
                                                 + e * sin_theta * xi / ((2.0 - tmp1)))
                                      - sin_theta * cos_theta
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                   * (4.0 * epsilon * sin_theta * ipow(cos_theta, 2)
                                              / ipow((epsilon * (epsilon + 2.0 * r * cos_theta)
                                                      + 1.0),
                                                     2)
                                      + 2.0
                                                * ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                           / (ipow((2.0 - tmp1), 2) * tmp1)
                                                   + e * cos_theta * xi / ((2.0 - tmp1)))
                                                * ((-e) * (epsilon * epsilon) * r * sin_theta
                                                           * ipow(cos_theta, 2) * xi
                                                           / (ipow(2.0 - tmp1, 2)
                                                              * Kokkos::pow(
                                                                      (epsilon
                                                                               * (epsilon
                                                                                  + 2.0 * r * cos_theta)
                                                                       + 1.0),
                                                                      (3.0 / 2.0)))
                                                   + 2.0 * e * (epsilon * epsilon) * r * sin_theta
                                                             * ipow(cos_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 3)
                                                                * (epsilon
                                                                           * (epsilon
                                                                              + 2.0 * r * cos_theta)
                                                                   + 1.0))
                                                   + 2.0 * e * epsilon * sin_theta * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1))
                                      + 2.0
                                                * (e * epsilon * r * sin_theta * cos_theta * xi
                                                           / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * sin_theta * xi / ((2.0 - tmp1)))
                                                * (e * (epsilon * epsilon) * r * ipow(sin_theta, 2)
                                                           * cos_theta * xi
                                                           / (ipow(2.0 - tmp1, 2)
                                                              * Kokkos::pow(
                                                                      (epsilon
                                                                               * (epsilon
                                                                                  + 2.0 * r * cos_theta)
                                                                       + 1.0),
                                                                      (3.0 / 2.0)))
                                                   - 2.0 * e * (epsilon * epsilon) * r
                                                             * ipow(sin_theta, 2) * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 3)
                                                                * (epsilon
                                                                           * (epsilon
                                                                              + 2.0 * r * cos_theta)
                                                                   + 1.0))
                                                   - e * epsilon * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * epsilon * ipow(cos_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)))
                           - 1.0 / 2.0
                                     * ((-2.0) * epsilon * ipow(cos_theta, 3)
                                                / ipow((epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0),
                                                       2)
                                        + (e * epsilon * r * sin_theta * cos_theta * xi
                                                   / (ipow(2.0 - tmp1, 2) * tmp1)
                                           + e * sin_theta * xi / ((2.0 - tmp1)))
                                                  * ((-2.0) * e * (epsilon * epsilon) * r
                                                             * sin_theta * ipow(cos_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2)
                                                                * Kokkos::pow(
                                                                        (epsilon
                                                                                 * (epsilon
                                                                                    + 2.0 * r * cos_theta)
                                                                         + 1.0),
                                                                        (3.0 / 2.0)))
                                                     + 4.0 * e * (epsilon * epsilon) * r * sin_theta
                                                               * ipow(cos_theta, 2) * xi
                                                               / (ipow(2.0 - tmp1, 3)
                                                                  * (epsilon
                                                                             * (epsilon
                                                                                + 2.0 * r * cos_theta)
                                                                     + 1.0))
                                                     + 4.0 * e * epsilon * sin_theta * cos_theta
                                                               * xi / (ipow(2.0 - tmp1, 2) * tmp1)))
                                     * (ipow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                                              + e * cos_theta * xi / ((2.0 - tmp1))),
                                             2)
                                        + ipow(sin_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                           - 1.0 / 2.0
                                     * ((-2.0) * epsilon * ipow(sin_theta, 2) * cos_theta
                                                / ipow((epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0),
                                                       2)
                                        + ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                   / (ipow(2.0 - tmp1, 2) * tmp1)
                                           + e * cos_theta * xi / ((2.0 - tmp1)))
                                                  * (2.0 * e * (epsilon * epsilon) * r
                                                             * ipow(sin_theta, 2) * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2)
                                                                * Kokkos::pow(
                                                                        (epsilon
                                                                                 * (epsilon
                                                                                    + 2.0 * r * cos_theta)
                                                                         + 1.0),
                                                                        (3.0 / 2.0)))
                                                     - 4.0 * e * (epsilon * epsilon) * r
                                                               * ipow(sin_theta, 2) * cos_theta * xi
                                                               / (ipow(2.0 - tmp1, 3)
                                                                  * (epsilon
                                                                             * (epsilon
                                                                                + 2.0 * r * cos_theta)
                                                                     + 1.0))
                                                     - 2.0 * e * epsilon * ipow(sin_theta, 2) * xi
                                                               / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + 2.0 * e * epsilon * ipow(cos_theta, 2) * xi
                                                               / (ipow(2.0 - tmp1, 2) * tmp1)))
                                     * (Kokkos::
                                                pow((e * epsilon * r * sin_theta * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * sin_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(cos_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0)))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::
                                pow(((-Kokkos::
                                              pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                            / (ipow(2.0 - tmp1, 2) * tmp1)
                                                    + e * cos_theta * xi / ((2.0 - tmp1)))
                                                           * (e * epsilon * r * sin_theta
                                                                      * cos_theta * xi
                                                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                                                              + e * sin_theta * xi / ((2.0 - tmp1)))
                                                   - sin_theta * cos_theta
                                                             / (epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0)),
                                                  2.0))
                                     + (Kokkos::
                                                pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * cos_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(sin_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                                               * (Kokkos::
                                                          pow((e * epsilon * r * sin_theta
                                                                       * cos_theta * xi
                                                                       / (ipow(2.0 - tmp1, 2)
                                                                          * tmp1)
                                                               + e * sin_theta * xi
                                                                         / ((2.0 - tmp1))),
                                                              2.0)
                                                  + ipow(cos_theta, 2)
                                                            / (epsilon
                                                                       * (epsilon
                                                                          + 2.0 * r * cos_theta)
                                                               + 1.0))),
                                    (3.0 / 2.0))
              - r
                        * (((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                    / (ipow(2.0 - tmp1, 2) * tmp1)
                            + e * cos_theta * xi / ((2.0 - tmp1)))
                                   * (e * epsilon * r * sin_theta * cos_theta * xi
                                              / (ipow(2.0 - tmp1, 2) * tmp1)
                                      + e * sin_theta * xi / ((2.0 - tmp1)))
                           - sin_theta * cos_theta
                                     / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (0.8192 * M_PI * epsilon * minus_pow_6 * plus_pow_6 * sin_theta
                                   * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::sin(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                   * cos_theta
                                   / Kokkos::
                                           pow((epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                                               (3.0 / 2.0))
                           - 0.4096 * minus_pow_6 * plus_pow_6
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           + 0.4096 * minus_pow_6 * plus_pow_6
                                     * (2.0 * M_PI * e * (epsilon * epsilon) * r
                                                * ipow(sin_theta, 2) * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        - 4.0 * M_PI * e * (epsilon * epsilon) * r
                                                  * ipow(sin_theta, 2) * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        - 2.0 * M_PI * e * epsilon * ipow(sin_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * epsilon * ipow(cos_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 1.6384 * (M_PI * M_PI) * minus_pow_6 * plus_pow_6 * sin_theta
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)
                           + 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon) * cos_theta
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           - 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                                     * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                     * sin_theta * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           + 2.4576 * minus_pow_6 * ipow(r + 1.0, 5)
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 4.9152 * M_PI * minus_pow_6 * ipow(r + 1.0, 5) * sin_theta
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           - 2.4576 * ipow(1.0 - r, 5) * plus_pow_6
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 4.9152 * M_PI * ipow(1.0 - r, 5) * plus_pow_6 * sin_theta
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1)
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              + r
                        * ((-2.0) * epsilon * ipow(sin_theta, 2) * cos_theta
                                   / Kokkos::
                                           pow((epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                                               2.0)
                           + ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                              + e * cos_theta * xi / ((2.0 - tmp1)))
                                     * (2.0 * e * (epsilon * epsilon) * r * ipow(sin_theta, 2)
                                                * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        - 4.0 * e * (epsilon * epsilon) * r * ipow(sin_theta, 2)
                                                  * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        - 2.0 * e * epsilon * ipow(sin_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * e * epsilon * ipow(cos_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)))
                        * (0.4096 * minus_pow_6 * plus_pow_6
                                   * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                              / (ipow(2.0 - tmp1, 2) * tmp1)
                                      + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::cos(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta / tmp1
                           + 2.4576 * minus_pow_6 * ipow(r + 1.0, 5)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           - 2.4576 * ipow(1.0 - r, 5) * plus_pow_6
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              - r
                        * (0.4096 * minus_pow_6 * plus_pow_6
                                   * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                              / (ipow(2.0 - tmp1, 2) * tmp1)
                                      + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::cos(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1)
                        * (2.0 * epsilon * sin_theta * ipow(cos_theta, 2)
                                   / Kokkos::
                                           pow((epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                                               2.0)
                           + ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                              + e * cos_theta * xi / ((2.0 - tmp1)))
                                     * ((-e) * (epsilon * epsilon) * r * sin_theta
                                                * ipow(cos_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        + 2.0 * e * (epsilon * epsilon) * r * sin_theta
                                                  * ipow(cos_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        + 2.0 * e * epsilon * sin_theta * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1))
                           + (e * epsilon * r * sin_theta * cos_theta * xi
                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                              + e * sin_theta * xi / ((2.0 - tmp1)))
                                     * (e * (epsilon * epsilon) * r * ipow(sin_theta, 2) * cos_theta
                                                * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        - 2.0 * e * (epsilon * epsilon) * r * ipow(sin_theta, 2)
                                                  * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        - e * epsilon * ipow(sin_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + e * epsilon * ipow(cos_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              + r
                        * (Kokkos::
                                   pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + e * cos_theta * xi / ((2.0 - tmp1))),
                                       2.0)
                           + ipow(sin_theta, 2) / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (20.0 * Kokkos::pow(tanh_term, 2.0) - 20.0)
                        * (0.4096 * minus_pow_6 * plus_pow_6
                                   * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                              / (ipow(2.0 - tmp1, 2) * tmp1)
                                      + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::cos(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta / tmp1
                           + 2.4576 * minus_pow_6 * ipow(r + 1.0, 5)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           - 2.4576 * ipow(1.0 - r, 5) * plus_pow_6
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              + r
                        * (Kokkos::
                                   pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + e * cos_theta * xi / ((2.0 - tmp1))),
                                       2.0)
                           + ipow(sin_theta, 2) / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * ((-0.8192) * M_PI * epsilon * minus_pow_6 * plus_pow_6
                                   * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::sin(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                   * ipow(cos_theta, 2)
                                   / Kokkos::
                                           pow((epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                                               (3.0 / 2.0))
                           - 0.4096 * minus_pow_6 * plus_pow_6
                                     * Kokkos::
                                             pow((2.0 * M_PI * e * epsilon * r * sin_theta
                                                          * cos_theta * xi
                                                          / (ipow(2.0 - tmp1, 2) * tmp1)
                                                  + 2.0 * M_PI * e * sin_theta * xi
                                                            / ((2.0 - tmp1))),
                                                 2.0)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           + 0.4096 * minus_pow_6 * plus_pow_6
                                     * ((-2.0) * M_PI * e * (epsilon * epsilon) * r * sin_theta
                                                * ipow(cos_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        + 4.0 * M_PI * e * (epsilon * epsilon) * r * sin_theta
                                                  * ipow(cos_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        + 4.0 * M_PI * e * epsilon * sin_theta * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 1.6384 * (M_PI * M_PI) * minus_pow_6 * plus_pow_6
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * ipow(cos_theta, 2)
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)
                           + 1.6384 * M_PI * minus_pow_6 * plus_pow_6
                                     * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon) * cos_theta
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           + 4.9152 * minus_pow_6 * ipow(r + 1.0, 5)
                                     * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 9.8304 * M_PI * minus_pow_6 * ipow(r + 1.0, 5)
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta / tmp1
                           + 12.288 * minus_pow_6 * ipow(r + 1.0, 4)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           - 4.9152 * ipow(1.0 - r, 5) * plus_pow_6
                                     * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 9.8304 * M_PI * ipow(1.0 - r, 5) * plus_pow_6
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta / tmp1
                           - 29.4912 * ipow(1.0 - r, 5) * ipow(r + 1.0, 5)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           + 12.288 * ipow(1.0 - r, 4) * plus_pow_6
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              + r
                        * (Kokkos::
                                   pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + e * cos_theta * xi / ((2.0 - tmp1))),
                                       2.0)
                           + ipow(sin_theta, 2) / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (1.0 / 2.0
                                   * (((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                               / (ipow(2.0 - tmp1, 2) * tmp1)
                                       + e * cos_theta * xi / ((2.0 - tmp1)))
                                              * (e * epsilon * r * sin_theta * cos_theta * xi
                                                         / (ipow(2.0 - tmp1, 2) * tmp1)
                                                 + e * sin_theta * xi / ((2.0 - tmp1)))
                                      - sin_theta * cos_theta
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                   * (4.0 * epsilon * sin_theta * ipow(cos_theta, 2)
                                              / Kokkos::
                                                      pow((epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0),
                                                          2.0)
                                      + 2.0
                                                * ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                           / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * cos_theta * xi / ((2.0 - tmp1)))
                                                * ((-e) * (epsilon * epsilon) * r * sin_theta
                                                           * ipow(cos_theta, 2) * xi
                                                           / (ipow(2.0 - tmp1, 2)
                                                              * Kokkos::pow(
                                                                      (epsilon
                                                                               * (epsilon
                                                                                  + 2.0 * r * cos_theta)
                                                                       + 1.0),
                                                                      (3.0 / 2.0)))
                                                   + 2.0 * e * (epsilon * epsilon) * r * sin_theta
                                                             * ipow(cos_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 3)
                                                                * (epsilon
                                                                           * (epsilon
                                                                              + 2.0 * r * cos_theta)
                                                                   + 1.0))
                                                   + 2.0 * e * epsilon * sin_theta * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1))
                                      + 2.0
                                                * (e * epsilon * r * sin_theta * cos_theta * xi
                                                           / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * sin_theta * xi / ((2.0 - tmp1)))
                                                * (e * (epsilon * epsilon) * r * ipow(sin_theta, 2)
                                                           * cos_theta * xi
                                                           / (ipow(2.0 - tmp1, 2)
                                                              * Kokkos::pow(
                                                                      (epsilon
                                                                               * (epsilon
                                                                                  + 2.0 * r * cos_theta)
                                                                       + 1.0),
                                                                      (3.0 / 2.0)))
                                                   - 2.0 * e * (epsilon * epsilon) * r
                                                             * ipow(sin_theta, 2) * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 3)
                                                                * (epsilon
                                                                           * (epsilon
                                                                              + 2.0 * r * cos_theta)
                                                                   + 1.0))
                                                   - e * epsilon * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * epsilon * ipow(cos_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)))
                           - 1.0 / 2.0
                                     * ((-2.0) * epsilon * ipow(cos_theta, 3)
                                                / Kokkos::
                                                        pow((epsilon
                                                                     * (epsilon
                                                                        + 2.0 * r * cos_theta)
                                                             + 1.0),
                                                            2.0)
                                        + (e * epsilon * r * sin_theta * cos_theta * xi
                                                   / (ipow(2.0 - tmp1, 2) * tmp1)
                                           + e * sin_theta * xi / ((2.0 - tmp1)))
                                                  * ((-2.0) * e * (epsilon * epsilon) * r
                                                             * sin_theta * ipow(cos_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2)
                                                                * Kokkos::pow(
                                                                        (epsilon
                                                                                 * (epsilon
                                                                                    + 2.0 * r * cos_theta)
                                                                         + 1.0),
                                                                        (3.0 / 2.0)))
                                                     + 4.0 * e * (epsilon * epsilon) * r * sin_theta
                                                               * ipow(cos_theta, 2) * xi
                                                               / (ipow(2.0 - tmp1, 3)
                                                                  * (epsilon
                                                                             * (epsilon
                                                                                + 2.0 * r * cos_theta)
                                                                     + 1.0))
                                                     + 4.0 * e * epsilon * sin_theta * cos_theta
                                                               * xi / (ipow(2.0 - tmp1, 2) * tmp1)))
                                     * (Kokkos::
                                                pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * cos_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(sin_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                           - 1.0 / 2.0
                                     * ((-2.0) * epsilon * ipow(sin_theta, 2) * cos_theta
                                                / Kokkos::
                                                        pow((epsilon
                                                                     * (epsilon
                                                                        + 2.0 * r * cos_theta)
                                                             + 1.0),
                                                            2.0)
                                        + ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                   / (ipow(2.0 - tmp1, 2) * tmp1)
                                           + e * cos_theta * xi / ((2.0 - tmp1)))
                                                  * (2.0 * e * (epsilon * epsilon) * r
                                                             * ipow(sin_theta, 2) * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2)
                                                                * Kokkos::pow(
                                                                        (epsilon
                                                                                 * (epsilon
                                                                                    + 2.0 * r * cos_theta)
                                                                         + 1.0),
                                                                        (3.0 / 2.0)))
                                                     - 4.0 * e * (epsilon * epsilon) * r
                                                               * ipow(sin_theta, 2) * cos_theta * xi
                                                               / (ipow(2.0 - tmp1, 3)
                                                                  * (epsilon
                                                                             * (epsilon
                                                                                + 2.0 * r * cos_theta)
                                                                     + 1.0))
                                                     - 2.0 * e * epsilon * ipow(sin_theta, 2) * xi
                                                               / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + 2.0 * e * epsilon * ipow(cos_theta, 2) * xi
                                                               / (ipow(2.0 - tmp1, 2) * tmp1)))
                                     * (Kokkos::
                                                pow((e * epsilon * r * sin_theta * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * sin_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(cos_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0)))
                        * (0.4096 * minus_pow_6 * plus_pow_6
                                   * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                              / (ipow(2.0 - tmp1, 2) * tmp1)
                                      + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::cos(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta / tmp1
                           + 2.4576 * minus_pow_6 * ipow(r + 1.0, 5)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           - 2.4576 * ipow(1.0 - r, 5) * plus_pow_6
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::
                                pow(((-Kokkos::
                                              pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                            / (ipow(2.0 - tmp1, 2) * tmp1)
                                                    + e * cos_theta * xi / ((2.0 - tmp1)))
                                                           * (e * epsilon * r * sin_theta
                                                                      * cos_theta * xi
                                                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                                                              + e * sin_theta * xi / ((2.0 - tmp1)))
                                                   - sin_theta * cos_theta
                                                             / (epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0)),
                                                  2.0))
                                     + (Kokkos::
                                                pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * cos_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(sin_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                                               * (Kokkos::
                                                          pow((e * epsilon * r * sin_theta
                                                                       * cos_theta * xi
                                                                       / (ipow(2.0 - tmp1, 2)
                                                                          * tmp1)
                                                               + e * sin_theta * xi
                                                                         / ((2.0 - tmp1))),
                                                              2.0)
                                                  + ipow(cos_theta, 2)
                                                            / (epsilon
                                                                       * (epsilon
                                                                          + 2.0 * r * cos_theta)
                                                               + 1.0))),
                                    (3.0 / 2.0))
              - (((-e) * epsilon * r * ipow(sin_theta, 2) * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                  + e * cos_theta * xi / ((2.0 - tmp1)))
                         * (e * epsilon * r * sin_theta * cos_theta * xi
                                    / (ipow(2.0 - tmp1, 2) * tmp1)
                            + e * sin_theta * xi / ((2.0 - tmp1)))
                 - sin_theta * cos_theta / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (0.4096 * minus_pow_6 * plus_pow_6
                                   * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                              / (ipow(2.0 - tmp1, 2) * tmp1)
                                      + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::cos(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1)
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              - (((-e) * epsilon * r * ipow(sin_theta, 2) * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                  + e * cos_theta * xi / ((2.0 - tmp1)))
                         * (e * epsilon * r * sin_theta * cos_theta * xi
                                    / (ipow(2.0 - tmp1, 2) * tmp1)
                            + e * sin_theta * xi / ((2.0 - tmp1)))
                 - sin_theta * cos_theta / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (0.8192 * M_PI * epsilon * r * minus_pow_6 * plus_pow_6 * sin_theta
                                   * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::sin(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                   * cos_theta
                                   / Kokkos::
                                           pow((epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                                               (3.0 / 2.0))
                           - 0.4096 * r * minus_pow_6 * plus_pow_6
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           + 1.6384 * (M_PI * M_PI) * r * minus_pow_6 * plus_pow_6 * sin_theta
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)
                           + 0.8192 * M_PI * r * minus_pow_6 * plus_pow_6
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon) * cos_theta
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           - 0.8192 * M_PI * r * minus_pow_6 * plus_pow_6
                                     * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                     * sin_theta * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           + 2.4576 * r * minus_pow_6 * ipow(r + 1.0, 5)
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 4.9152 * M_PI * r * minus_pow_6 * ipow(r + 1.0, 5) * sin_theta
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           - 2.4576 * r * ipow(1.0 - r, 5) * plus_pow_6
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 4.9152 * M_PI * r * ipow(1.0 - r, 5) * plus_pow_6 * sin_theta
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           + 0.4096 * minus_pow_6 * plus_pow_6
                                     * (2.0 * M_PI * e * (epsilon * epsilon) * (r * r)
                                                * ipow(sin_theta, 2) * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        - 4.0 * M_PI * e * (epsilon * epsilon) * (r * r)
                                                  * ipow(sin_theta, 2) * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        - 4.0 * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * epsilon * r * ipow(cos_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1)
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              - (((-e) * epsilon * r * ipow(sin_theta, 2) * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                  + e * cos_theta * xi / ((2.0 - tmp1)))
                         * (e * epsilon * r * sin_theta * cos_theta * xi
                                    / (ipow(2.0 - tmp1, 2) * tmp1)
                            + e * sin_theta * xi / ((2.0 - tmp1)))
                 - sin_theta * cos_theta / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (1.0 / 2.0
                                   * (((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                               / (ipow(2.0 - tmp1, 2) * tmp1)
                                       + e * cos_theta * xi / ((2.0 - tmp1)))
                                              * (e * epsilon * r * sin_theta * cos_theta * xi
                                                         / (ipow(2.0 - tmp1, 2) * tmp1)
                                                 + e * sin_theta * xi / ((2.0 - tmp1)))
                                      - sin_theta * cos_theta
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                   * ((-4.0) * epsilon * r * ipow(sin_theta, 2) * cos_theta
                                              / Kokkos::
                                                      pow((epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0),
                                                          2.0)
                                      + 2.0
                                                * ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                           / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * cos_theta * xi / ((2.0 - tmp1)))
                                                * (e * (epsilon * epsilon) * (r * r)
                                                           * ipow(sin_theta, 2) * cos_theta * xi
                                                           / (ipow(2.0 - tmp1, 2)
                                                              * Kokkos::pow(
                                                                      (epsilon
                                                                               * (epsilon
                                                                                  + 2.0 * r * cos_theta)
                                                                       + 1.0),
                                                                      (3.0 / 2.0)))
                                                   - 2.0 * e * (epsilon * epsilon) * (r * r)
                                                             * ipow(sin_theta, 2) * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 3)
                                                                * (epsilon
                                                                           * (epsilon
                                                                              + 2.0 * r * cos_theta)
                                                                   + 1.0))
                                                   - 2.0 * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * epsilon * r * ipow(cos_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * cos_theta * xi / ((2.0 - tmp1)))
                                      + 2.0
                                                * (e * epsilon * r * sin_theta * cos_theta * xi
                                                           / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * sin_theta * xi / ((2.0 - tmp1)))
                                                * ((-e) * (epsilon * epsilon) * (r * r)
                                                           * ipow(sin_theta, 3) * xi
                                                           / (ipow(2.0 - tmp1, 2)
                                                              * Kokkos::pow(
                                                                      (epsilon
                                                                               * (epsilon
                                                                                  + 2.0 * r * cos_theta)
                                                                       + 1.0),
                                                                      (3.0 / 2.0)))
                                                   + 2.0 * e * (epsilon * epsilon) * (r * r)
                                                             * ipow(sin_theta, 3) * xi
                                                             / (ipow(2.0 - tmp1, 3)
                                                                * (epsilon
                                                                           * (epsilon
                                                                              + 2.0 * r * cos_theta)
                                                                   + 1.0))
                                                   - 3.0 * e * epsilon * r * sin_theta * cos_theta
                                                             * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   - e * sin_theta * xi / ((2.0 - tmp1)))
                                      + 2.0 * ipow(sin_theta, 2)
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)
                                      - 2.0 * ipow(cos_theta, 2)
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                           - 1.0 / 2.0
                                     * (Kokkos::
                                                pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * cos_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(sin_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                                     * (2.0 * epsilon * r * sin_theta * ipow(cos_theta, 2)
                                                / Kokkos::
                                                        pow((epsilon
                                                                     * (epsilon
                                                                        + 2.0 * r * cos_theta)
                                                             + 1.0),
                                                            2.0)
                                        + (e * epsilon * r * sin_theta * cos_theta * xi
                                                   / (ipow(2.0 - tmp1, 2) * tmp1)
                                           + e * sin_theta * xi / ((2.0 - tmp1)))
                                                  * (2.0 * e * (epsilon * epsilon) * (r * r)
                                                             * ipow(sin_theta, 2) * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2)
                                                                * Kokkos::pow(
                                                                        (epsilon
                                                                                 * (epsilon
                                                                                    + 2.0 * r * cos_theta)
                                                                         + 1.0),
                                                                        (3.0 / 2.0)))
                                                     - 4.0 * e * (epsilon * epsilon) * (r * r)
                                                               * ipow(sin_theta, 2) * cos_theta * xi
                                                               / (ipow(2.0 - tmp1, 3)
                                                                  * (epsilon
                                                                             * (epsilon
                                                                                + 2.0 * r * cos_theta)
                                                                     + 1.0))
                                                     - 4.0 * e * epsilon * r * ipow(sin_theta, 2)
                                                               * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + 2.0 * e * epsilon * r * ipow(cos_theta, 2)
                                                               * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + 2.0 * e * cos_theta * xi / ((2.0 - tmp1)))
                                        - 2.0 * sin_theta * cos_theta
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                           - 1.0 / 2.0
                                     * (Kokkos::
                                                pow((e * epsilon * r * sin_theta * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * sin_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(cos_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                                     * (2.0 * epsilon * r * ipow(sin_theta, 3)
                                                / Kokkos::
                                                        pow((epsilon
                                                                     * (epsilon
                                                                        + 2.0 * r * cos_theta)
                                                             + 1.0),
                                                            2.0)
                                        + ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                   / (ipow(2.0 - tmp1, 2) * tmp1)
                                           + e * cos_theta * xi / ((2.0 - tmp1)))
                                                  * ((-2.0) * e * (epsilon * epsilon) * (r * r)
                                                             * ipow(sin_theta, 3) * xi
                                                             / (ipow(2.0 - tmp1, 2)
                                                                * Kokkos::pow(
                                                                        (epsilon
                                                                                 * (epsilon
                                                                                    + 2.0 * r * cos_theta)
                                                                         + 1.0),
                                                                        (3.0 / 2.0)))
                                                     + 4.0 * e * (epsilon * epsilon) * (r * r)
                                                               * ipow(sin_theta, 3) * xi
                                                               / (ipow(2.0 - tmp1, 3)
                                                                  * (epsilon
                                                                             * (epsilon
                                                                                + 2.0 * r * cos_theta)
                                                                     + 1.0))
                                                     - 6.0 * e * epsilon * r * sin_theta * cos_theta
                                                               * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     - 2.0 * e * sin_theta * xi / ((2.0 - tmp1)))
                                        + 2.0 * sin_theta * cos_theta
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0)))
                        * (0.4096 * minus_pow_6 * plus_pow_6
                                   * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                              / (ipow(2.0 - tmp1, 2) * tmp1)
                                      + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::cos(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta / tmp1
                           + 2.4576 * minus_pow_6 * ipow(r + 1.0, 5)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           - 2.4576 * ipow(1.0 - r, 5) * plus_pow_6
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::
                                pow(((-Kokkos::
                                              pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                            / (ipow(2.0 - tmp1, 2) * tmp1)
                                                    + e * cos_theta * xi / ((2.0 - tmp1)))
                                                           * (e * epsilon * r * sin_theta
                                                                      * cos_theta * xi
                                                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                                                              + e * sin_theta * xi / ((2.0 - tmp1)))
                                                   - sin_theta * cos_theta
                                                             / (epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0)),
                                                  2.0))
                                     + (Kokkos::
                                                pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * cos_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(sin_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                                               * (Kokkos::
                                                          pow((e * epsilon * r * sin_theta
                                                                       * cos_theta * xi
                                                                       / (ipow(2.0 - tmp1, 2)
                                                                          * tmp1)
                                                               + e * sin_theta * xi
                                                                         / ((2.0 - tmp1))),
                                                              2.0)
                                                  + ipow(cos_theta, 2)
                                                            / (epsilon
                                                                       * (epsilon
                                                                          + 2.0 * r * cos_theta)
                                                               + 1.0))),
                                    (3.0 / 2.0))
              + (0.4096 * minus_pow_6 * plus_pow_6
                         * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                    / (ipow(2.0 - tmp1, 2) * tmp1)
                            + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                         * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                         * Kokkos::cos(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                 - 0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta
                           * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           * Kokkos::sin(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           / tmp1)
                        * (2.0 * epsilon * r * sin_theta * ipow(cos_theta, 2)
                                   / Kokkos::
                                           pow((epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                                               2.0)
                           + (e * epsilon * r * sin_theta * cos_theta * xi
                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                              + e * sin_theta * xi / ((2.0 - tmp1)))
                                     * (2.0 * e * (epsilon * epsilon) * (r * r) * ipow(sin_theta, 2)
                                                * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        - 4.0 * e * (epsilon * epsilon) * (r * r)
                                                  * ipow(sin_theta, 2) * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        - 4.0 * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * e * epsilon * r * ipow(cos_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * e * cos_theta * xi / ((2.0 - tmp1)))
                           - 2.0 * sin_theta * cos_theta
                                     / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              + (0.4096 * minus_pow_6 * plus_pow_6
                         * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                    / (ipow(2.0 - tmp1, 2) * tmp1)
                            + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                         * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                         * Kokkos::cos(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                 - 0.8192 * M_PI * minus_pow_6 * plus_pow_6 * sin_theta
                           * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           * Kokkos::sin(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           / tmp1)
                        * (Kokkos::
                                   pow((e * epsilon * r * sin_theta * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + e * sin_theta * xi / ((2.0 - tmp1))),
                                       2.0)
                           + ipow(cos_theta, 2) / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (1.0 / 2.0
                                   * (((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                               / (ipow(2.0 - tmp1, 2) * tmp1)
                                       + e * cos_theta * xi / ((2.0 - tmp1)))
                                              * (e * epsilon * r * sin_theta * cos_theta * xi
                                                         / (ipow(2.0 - tmp1, 2) * tmp1)
                                                 + e * sin_theta * xi / ((2.0 - tmp1)))
                                      - sin_theta * cos_theta
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                   * ((-4.0) * epsilon * r * ipow(sin_theta, 2) * cos_theta
                                              / Kokkos::
                                                      pow((epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0),
                                                          2.0)
                                      + 2.0
                                                * ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                           / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * cos_theta * xi / ((2.0 - tmp1)))
                                                * (e * (epsilon * epsilon) * (r * r)
                                                           * ipow(sin_theta, 2) * cos_theta * xi
                                                           / (ipow(2.0 - tmp1, 2)
                                                              * Kokkos::pow(
                                                                      (epsilon
                                                                               * (epsilon
                                                                                  + 2.0 * r * cos_theta)
                                                                       + 1.0),
                                                                      (3.0 / 2.0)))
                                                   - 2.0 * e * (epsilon * epsilon) * (r * r)
                                                             * ipow(sin_theta, 2) * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 3)
                                                                * (epsilon
                                                                           * (epsilon
                                                                              + 2.0 * r * cos_theta)
                                                                   + 1.0))
                                                   - 2.0 * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * epsilon * r * ipow(cos_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * cos_theta * xi / ((2.0 - tmp1)))
                                      + 2.0
                                                * (e * epsilon * r * sin_theta * cos_theta * xi
                                                           / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   + e * sin_theta * xi / ((2.0 - tmp1)))
                                                * ((-e) * (epsilon * epsilon) * (r * r)
                                                           * ipow(sin_theta, 3) * xi
                                                           / (ipow(2.0 - tmp1, 2)
                                                              * Kokkos::pow(
                                                                      (epsilon
                                                                               * (epsilon
                                                                                  + 2.0 * r * cos_theta)
                                                                       + 1.0),
                                                                      (3.0 / 2.0)))
                                                   + 2.0 * e * (epsilon * epsilon) * (r * r)
                                                             * ipow(sin_theta, 3) * xi
                                                             / (ipow(2.0 - tmp1, 3)
                                                                * (epsilon
                                                                           * (epsilon
                                                                              + 2.0 * r * cos_theta)
                                                                   + 1.0))
                                                   - 3.0 * e * epsilon * r * sin_theta * cos_theta
                                                             * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                   - e * sin_theta * xi / ((2.0 - tmp1)))
                                      + 2.0 * ipow(sin_theta, 2)
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)
                                      - 2.0 * ipow(cos_theta, 2)
                                                / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                           - 1.0 / 2.0
                                     * (Kokkos::
                                                pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * cos_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(sin_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                                     * (2.0 * epsilon * r * sin_theta * ipow(cos_theta, 2)
                                                / Kokkos::
                                                        pow((epsilon
                                                                     * (epsilon
                                                                        + 2.0 * r * cos_theta)
                                                             + 1.0),
                                                            2.0)
                                        + (e * epsilon * r * sin_theta * cos_theta * xi
                                                   / (ipow(2.0 - tmp1, 2) * tmp1)
                                           + e * sin_theta * xi / ((2.0 - tmp1)))
                                                  * (2.0 * e * (epsilon * epsilon) * (r * r)
                                                             * ipow(sin_theta, 2) * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2)
                                                                * Kokkos::pow(
                                                                        (epsilon
                                                                                 * (epsilon
                                                                                    + 2.0 * r * cos_theta)
                                                                         + 1.0),
                                                                        (3.0 / 2.0)))
                                                     - 4.0 * e * (epsilon * epsilon) * (r * r)
                                                               * ipow(sin_theta, 2) * cos_theta * xi
                                                               / (ipow(2.0 - tmp1, 3)
                                                                  * (epsilon
                                                                             * (epsilon
                                                                                + 2.0 * r * cos_theta)
                                                                     + 1.0))
                                                     - 4.0 * e * epsilon * r * ipow(sin_theta, 2)
                                                               * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + 2.0 * e * epsilon * r * ipow(cos_theta, 2)
                                                               * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + 2.0 * e * cos_theta * xi / ((2.0 - tmp1)))
                                        - 2.0 * sin_theta * cos_theta
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                           - 1.0 / 2.0
                                     * (Kokkos::
                                                pow((e * epsilon * r * sin_theta * cos_theta * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * sin_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(cos_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                                     * (2.0 * epsilon * r * ipow(sin_theta, 3)
                                                / Kokkos::
                                                        pow((epsilon
                                                                     * (epsilon
                                                                        + 2.0 * r * cos_theta)
                                                             + 1.0),
                                                            2.0)
                                        + ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                   / (ipow(2.0 - tmp1, 2) * tmp1)
                                           + e * cos_theta * xi / ((2.0 - tmp1)))
                                                  * ((-2.0) * e * (epsilon * epsilon) * (r * r)
                                                             * ipow(sin_theta, 3) * xi
                                                             / (ipow(2.0 - tmp1, 2)
                                                                * Kokkos::pow(
                                                                        (epsilon
                                                                                 * (epsilon
                                                                                    + 2.0 * r * cos_theta)
                                                                         + 1.0),
                                                                        (3.0 / 2.0)))
                                                     + 4.0 * e * (epsilon * epsilon) * (r * r)
                                                               * ipow(sin_theta, 3) * xi
                                                               / (ipow(2.0 - tmp1, 3)
                                                                  * (epsilon
                                                                             * (epsilon
                                                                                + 2.0 * r * cos_theta)
                                                                     + 1.0))
                                                     - 6.0 * e * epsilon * r * sin_theta * cos_theta
                                                               * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     - 2.0 * e * sin_theta * xi / ((2.0 - tmp1)))
                                        + 2.0 * sin_theta * cos_theta
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0)))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::
                                pow(((-Kokkos::
                                              pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                            / (ipow(2.0 - tmp1, 2) * tmp1)
                                                    + e * cos_theta * xi / ((2.0 - tmp1)))
                                                           * (e * epsilon * r * sin_theta
                                                                      * cos_theta * xi
                                                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                                                              + e * sin_theta * xi / ((2.0 - tmp1)))
                                                   - sin_theta * cos_theta
                                                             / (epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0)),
                                                  2.0))
                                     + (Kokkos::
                                                pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                             / (ipow(2.0 - tmp1, 2) * tmp1)
                                                     + e * cos_theta * xi / ((2.0 - tmp1))),
                                                    2.0)
                                        + ipow(sin_theta, 2)
                                                  / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                     + 1.0))
                                               * (Kokkos::
                                                          pow((e * epsilon * r * sin_theta
                                                                       * cos_theta * xi
                                                                       / (ipow(2.0 - tmp1, 2)
                                                                          * tmp1)
                                                               + e * sin_theta * xi
                                                                         / ((2.0 - tmp1))),
                                                              2.0)
                                                  + ipow(cos_theta, 2)
                                                            / (epsilon
                                                                       * (epsilon
                                                                          + 2.0 * r * cos_theta)
                                                               + 1.0))),
                                    (3.0 / 2.0))
              + (Kokkos::
                         pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                              + e * cos_theta * xi / ((2.0 - tmp1))),
                             2.0)
                 + ipow(sin_theta, 2) / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * (0.4096 * minus_pow_6 * plus_pow_6
                                   * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                              / (ipow(2.0 - tmp1, 2) * tmp1)
                                      + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                   * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::cos(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           + 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta / tmp1
                           + 2.4576 * minus_pow_6 * ipow(r + 1.0, 5)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           - 2.4576 * ipow(1.0 - r, 5) * plus_pow_6
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              + (Kokkos::
                         pow((e * epsilon * r * sin_theta * cos_theta * xi
                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                              + e * sin_theta * xi / ((2.0 - tmp1))),
                             2.0)
                 + ipow(cos_theta, 2) / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * ((-0.8192) * M_PI * epsilon * r * minus_pow_6 * plus_pow_6
                                   * ipow(sin_theta, 2)
                                   * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                   * Kokkos::sin(
                                           2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                   / Kokkos::
                                           pow((epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                                               (3.0 / 2.0))
                           - 0.4096 * r * minus_pow_6 * plus_pow_6
                                     * Kokkos::
                                             pow(((-2.0) * M_PI * e * epsilon * r
                                                          * ipow(sin_theta, 2) * xi
                                                          / (ipow(2.0 - tmp1, 2) * tmp1)
                                                  + 2.0 * M_PI * e * cos_theta * xi
                                                            / ((2.0 - tmp1))),
                                                 2.0)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           - 1.6384 * (M_PI * M_PI) * r * minus_pow_6 * plus_pow_6
                                     * ipow(sin_theta, 2)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)
                           - 1.6384 * M_PI * r * minus_pow_6 * plus_pow_6
                                     * ((-2.0) * M_PI * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + 2.0 * M_PI * e * cos_theta * xi / ((2.0 - tmp1)))
                                     * sin_theta * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     / tmp1
                           + 0.4096 * minus_pow_6 * plus_pow_6
                                     * ((-2.0) * M_PI * e * (epsilon * epsilon) * (r * r)
                                                * ipow(sin_theta, 3) * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        + 4.0 * M_PI * e * (epsilon * epsilon) * (r * r)
                                                  * ipow(sin_theta, 3) * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        - 6.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        - 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                                     * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::cos(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           - 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                                     * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                                     * Kokkos::sin(
                                             2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                                     * cos_theta / tmp1)
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0)))
              - (0.4096 * minus_pow_6 * plus_pow_6
                         * (2.0 * M_PI * e * epsilon * r * sin_theta * cos_theta * xi
                                    / (ipow(2.0 - tmp1, 2) * tmp1)
                            + 2.0 * M_PI * e * sin_theta * xi / ((2.0 - tmp1)))
                         * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                         * Kokkos::cos(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                 + 0.8192 * M_PI * minus_pow_6 * plus_pow_6
                           * Kokkos::sin(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                           * Kokkos::sin(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           * cos_theta / tmp1
                 + 2.4576 * minus_pow_6 * ipow(r + 1.0, 5)
                           * Kokkos::sin(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon)
                 - 2.4576 * ipow(1.0 - r, 5) * plus_pow_6
                           * Kokkos::sin(2.0 * M_PI * e * r * sin_theta * xi / ((2.0 - tmp1)))
                           * Kokkos::cos(2.0 * M_PI * (1.0 - tmp1) / epsilon))
                        * ((-2.0) * epsilon * r * ipow(sin_theta, 2) * cos_theta
                                   / Kokkos::
                                           pow((epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                                               2.0)
                           + ((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                              + e * cos_theta * xi / ((2.0 - tmp1)))
                                     * (e * (epsilon * epsilon) * (r * r) * ipow(sin_theta, 2)
                                                * cos_theta * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        - 2.0 * e * (epsilon * epsilon) * (r * r)
                                                  * ipow(sin_theta, 2) * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        - 2.0 * e * epsilon * r * ipow(sin_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + e * epsilon * r * ipow(cos_theta, 2) * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        + e * cos_theta * xi / ((2.0 - tmp1)))
                           + (e * epsilon * r * sin_theta * cos_theta * xi
                                      / (ipow(2.0 - tmp1, 2) * tmp1)
                              + e * sin_theta * xi / ((2.0 - tmp1)))
                                     * ((-e) * (epsilon * epsilon) * (r * r) * ipow(sin_theta, 3)
                                                * xi
                                                / (ipow(2.0 - tmp1, 2)
                                                   * Kokkos::
                                                           pow((epsilon
                                                                        * (epsilon
                                                                           + 2.0 * r * cos_theta)
                                                                + 1.0),
                                                               (3.0 / 2.0)))
                                        + 2.0 * e * (epsilon * epsilon) * (r * r)
                                                  * ipow(sin_theta, 3) * xi
                                                  / (ipow(2.0 - tmp1, 3)
                                                     * (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                        + 1.0))
                                        - 3.0 * e * epsilon * r * sin_theta * cos_theta * xi
                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                        - e * sin_theta * xi / ((2.0 - tmp1)))
                           + ipow(sin_theta, 2) / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)
                           - ipow(cos_theta, 2) / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                        * Kokkos::exp(-tanh_term)
                        / Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0))))
                     / (r
                        * Kokkos::sqrt(
                                (-Kokkos::
                                         pow((((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                       / (ipow(2.0 - tmp1, 2) * tmp1)
                                               + e * cos_theta * xi / ((2.0 - tmp1)))
                                                      * (e * epsilon * r * sin_theta * cos_theta
                                                                 * xi / (ipow(2.0 - tmp1, 2) * tmp1)
                                                         + e * sin_theta * xi / ((2.0 - tmp1)))
                                              - sin_theta * cos_theta
                                                        / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                           + 1.0)),
                                             2.0))
                                + (Kokkos::
                                           pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi
                                                        / (ipow(2.0 - tmp1, 2) * tmp1)
                                                + e * cos_theta * xi / ((2.0 - tmp1))),
                                               2.0)
                                   + ipow(sin_theta, 2)
                                             / (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0))
                                          * (Kokkos::
                                                     pow((e * epsilon * r * sin_theta * cos_theta
                                                                  * xi
                                                                  / (ipow(2.0 - tmp1, 2) * tmp1)
                                                          + e * sin_theta * xi / ((2.0 - tmp1))),
                                                         2.0)
                                             + ipow(cos_theta, 2)
                                                       / (epsilon * (epsilon + 2.0 * r * cos_theta)
                                                          + 1.0))));
}

template class ManufacturedRHS<CartesianSolution<CzarnyToCartesian<R, Theta, X, Y>>>;
