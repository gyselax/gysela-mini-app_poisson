// SPDX-License-Identifier: MIT
#include "rhs.hpp"

#include <math_tools.hpp>

template <>
KOKKOS_FUNCTION double
ManufacturedRHS<CurvilinearSolution<CzarnyToCartesian<R, Theta, X, Y>>>::
    solution_at_pole(Coord<R, Theta> const& coord) const {
    return 0.0;
}

template <>
KOKKOS_FUNCTION double
ManufacturedRHS<CurvilinearSolution<CzarnyToCartesian<R, Theta, X, Y>>>::
    non_singular_solution(Coord<R, Theta> const& coord) const {
    const double r = ddc::get<R>(coord);
    const double theta = ddc::get<Theta>(coord);
    const double epsilon = m_coordinate_converter.epsilon();
    const double e = m_coordinate_converter.e();

    const double sin_theta = Kokkos::sin(theta);
    const double cos_theta = Kokkos::cos(theta);

    const double tanh_term = Kokkos::tanh(20.0 * r - 14.0);
    const double coeff_alpha = Kokkos::exp(tanh_term);

    const double xi = 1. / Kokkos::sqrt(1. - epsilon * epsilon * 0.25);
    const double tmp1 =
        Kokkos::sqrt(epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0);

    return 0.4096 * ipow(r, 6) * ipow(r - 1.0, 6) * coeff_alpha *
               Kokkos::cos(11.0 * theta) -
           ipow(r, 4) *
               (4.5056 * r * ipow(r - 1.0, 6) *
                    (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * cos_theta * xi / ((2.0 - tmp1))) *
                         (e * epsilon * r * sin_theta * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          e * sin_theta * xi / ((2.0 - tmp1))) -
                     sin_theta * cos_theta /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    (20.0 * Kokkos::pow(tanh_term, 2.0) - 20.0) *
                    Kokkos::exp(-tanh_term) * Kokkos::sin(11.0 * theta) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) +
                4.5056 * r * ipow(r - 1.0, 6) *
                    (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * cos_theta * xi / ((2.0 - tmp1))) *
                         (e * epsilon * r * sin_theta * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          e * sin_theta * xi / ((2.0 - tmp1))) -
                     sin_theta * cos_theta /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    (1.0 / 2.0 *
                         (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * cos_theta * xi / ((2.0 - tmp1))) *
                              (e * epsilon * r * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * sin_theta * xi / ((2.0 - tmp1))) -
                          sin_theta * cos_theta /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                         (4.0 * epsilon * sin_theta * ipow(cos_theta, 2) /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          2.0 *
                              ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) *
                              ((-e) * (epsilon * epsilon) * r * sin_theta *
                                   ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) +
                               2.0 * e * (epsilon * epsilon) * r * sin_theta *
                                   ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) +
                               2.0 * e * epsilon * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1)) +
                          2.0 *
                              (e * epsilon * r * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * sin_theta * xi / ((2.0 - tmp1))) *
                              (e * (epsilon * epsilon) * r *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) -
                               2.0 * e * (epsilon * epsilon) * r *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               e * epsilon * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * epsilon * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1))) -
                     1.0 / 2.0 *
                         ((-2.0) * epsilon * ipow(cos_theta, 3) /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          (e * epsilon * r * sin_theta * cos_theta * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * sin_theta * xi / ((2.0 - tmp1))) *
                              ((-2.0) * e * (epsilon * epsilon) * r *
                                   sin_theta * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) +
                               4.0 * e * (epsilon * epsilon) * r * sin_theta *
                                   ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) +
                               4.0 * e * epsilon * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1))) *
                         (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * cos_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) -
                     1.0 / 2.0 *
                         ((-2.0) * epsilon * ipow(sin_theta, 2) * cos_theta /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * cos_theta * xi / ((2.0 - tmp1))) *
                              (2.0 * e * (epsilon * epsilon) * r *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) -
                               4.0 * e * (epsilon * epsilon) * r *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               2.0 * e * epsilon * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               2.0 * e * epsilon * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1))) *
                         (Kokkos::pow((e * epsilon * r * sin_theta * cos_theta *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * sin_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(cos_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0))) *
                    Kokkos::exp(-tanh_term) * Kokkos::sin(11.0 * theta) /
                    Kokkos::pow(
                        ((-Kokkos::pow(
                             (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) *
                                  (e * epsilon * r * sin_theta * cos_theta *
                                       xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                   e * sin_theta * xi / ((2.0 - tmp1))) -
                              sin_theta * cos_theta /
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0)),
                             2.0)) +
                         (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * cos_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                             (Kokkos::pow(
                                  (e * epsilon * r * sin_theta * cos_theta *
                                       xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                   e * sin_theta * xi / ((2.0 - tmp1))),
                                  2.0) +
                              ipow(cos_theta, 2) /
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0))),
                        (3.0 / 2.0)) +
                4.5056 * r * ipow(r - 1.0, 6) *
                    (2.0 * epsilon * sin_theta * ipow(cos_theta, 2) /
                         Kokkos::pow(
                             (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                             2.0) +
                     ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * cos_theta * xi / ((2.0 - tmp1))) *
                         ((-e) * (epsilon * epsilon) * r * sin_theta *
                              ipow(cos_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) *
                               Kokkos::pow(
                                   (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                    1.0),
                                   (3.0 / 2.0))) +
                          2.0 * e * (epsilon * epsilon) * r * sin_theta *
                              ipow(cos_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 3) *
                               (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                1.0)) +
                          2.0 * e * epsilon * sin_theta * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1)) +
                     (e * epsilon * r * sin_theta * cos_theta * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * sin_theta * xi / ((2.0 - tmp1))) *
                         (e * (epsilon * epsilon) * r * ipow(sin_theta, 2) *
                              cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) *
                               Kokkos::pow(
                                   (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                    1.0),
                                   (3.0 / 2.0))) -
                          2.0 * e * (epsilon * epsilon) * r *
                              ipow(sin_theta, 2) * cos_theta * xi /
                              (ipow(2.0 - tmp1, 3) *
                               (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                1.0)) -
                          e * epsilon * ipow(sin_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          e * epsilon * ipow(cos_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1))) *
                    Kokkos::exp(-tanh_term) * Kokkos::sin(11.0 * theta) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) +
                27.0336 * r * ipow(r - 1.0, 5) *
                    (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * cos_theta * xi / ((2.0 - tmp1))) *
                         (e * epsilon * r * sin_theta * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          e * sin_theta * xi / ((2.0 - tmp1))) -
                     sin_theta * cos_theta /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    Kokkos::exp(-tanh_term) * Kokkos::sin(11.0 * theta) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) +
                r * (12.288 * r * ipow(r - 1.0, 4) * Kokkos::cos(11.0 * theta) + 17.2032 * ipow(r - 1.0, 5) * Kokkos::cos(11.0 * theta)) *
                    (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * cos_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                     ipow(sin_theta, 2) /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    Kokkos::exp(-tanh_term) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta /
                                  (Kokkos::sqrt(1.0 - 1.0 / 4.0 *
                                                          (epsilon * epsilon)) *
                                   (2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(
                             ((-e) * epsilon * r * ipow(sin_theta, 2) /
                                  (Kokkos::sqrt(1.0 - 1.0 / 4.0 *
                                                          (epsilon * epsilon)) *
                                   ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta /
                                  (Kokkos::sqrt(1.0 - 1.0 / 4.0 *
                                                          (epsilon * epsilon)) *
                                   (2.0 - tmp1))),
                             2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta /
                                      (Kokkos::sqrt(1.0 -
                                                    1.0 / 4.0 *
                                                        (epsilon * epsilon)) *
                                       ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) +
                r * (2.4576 * r * ipow(r - 1.0, 5) * Kokkos::cos(11.0 * theta) + 2.4576 * ipow(r - 1.0, 6) * Kokkos::cos(11.0 * theta)) *
                    ((-2.0) * epsilon * ipow(sin_theta, 2) * cos_theta /
                         Kokkos::pow(
                             (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                             2.0) +
                     ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * cos_theta * xi / ((2.0 - tmp1))) *
                         (2.0 * e * (epsilon * epsilon) * r *
                              ipow(sin_theta, 2) * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) *
                               Kokkos::pow(
                                   (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                    1.0),
                                   (3.0 / 2.0))) -
                          4.0 * e * (epsilon * epsilon) * r *
                              ipow(sin_theta, 2) * cos_theta * xi /
                              (ipow(2.0 - tmp1, 3) *
                               (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                1.0)) -
                          2.0 * e * epsilon * ipow(sin_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          2.0 * e * epsilon * ipow(cos_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1))) *
                    Kokkos::exp(-tanh_term) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) +
                r * (2.4576 * r * ipow(r - 1.0, 5) * Kokkos::cos(11.0 * theta) + 2.4576 * ipow(r - 1.0, 6) * Kokkos::cos(11.0 * theta)) *
                    (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * cos_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                     ipow(sin_theta, 2) /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    (20.0 * Kokkos::pow(tanh_term, 2.0) - 20.0) *
                    Kokkos::exp(-tanh_term) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) /
                                  (Kokkos::sqrt(1.0 - 1.0 / 4.0 *
                                                          (epsilon * epsilon)) *
                                   ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) +
                r * (2.4576 * r * ipow(r - 1.0, 5) * Kokkos::cos(11.0 * theta) + 2.4576 * ipow(r - 1.0, 6) * Kokkos::cos(11.0 * theta)) *
                    (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * cos_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                     ipow(sin_theta, 2) /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    (1.0 / 2.0 *
                         (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * cos_theta * xi / ((2.0 - tmp1))) *
                              (e * epsilon * r * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * sin_theta * xi / ((2.0 - tmp1))) -
                          sin_theta * cos_theta /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                         (4.0 * epsilon * sin_theta * ipow(cos_theta, 2) /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          2.0 *
                              ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) *
                              ((-e) * (epsilon * epsilon) * r * sin_theta *
                                   ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) +
                               2.0 * e * (epsilon * epsilon) * r * sin_theta *
                                   ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) +
                               2.0 * e * epsilon * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1)) +
                          2.0 *
                              (e * epsilon * r * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * sin_theta * xi / ((2.0 - tmp1))) *
                              (e * (epsilon * epsilon) * r *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) -
                               2.0 * e * (epsilon * epsilon) * r *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               e * epsilon * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * epsilon * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1))) -
                     1.0 / 2.0 *
                         ((-2.0) * epsilon * ipow(cos_theta, 3) /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          (e * epsilon * r * sin_theta * cos_theta * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * sin_theta * xi / ((2.0 - tmp1))) *
                              ((-2.0) * e * (epsilon * epsilon) * r *
                                   sin_theta * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) +
                               4.0 * e * (epsilon * epsilon) * r * sin_theta *
                                   ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) +
                               4.0 * e * epsilon * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1))) *
                         (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * cos_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) -
                     1.0 / 2.0 *
                         ((-2.0) * epsilon * ipow(sin_theta, 2) * cos_theta /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * cos_theta * xi / ((2.0 - tmp1))) *
                              (2.0 * e * (epsilon * epsilon) * r *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) -
                               4.0 * e * (epsilon * epsilon) * r *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               2.0 * e * epsilon * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               2.0 * e * epsilon * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1))) *
                         (Kokkos::pow((e * epsilon * r * sin_theta * cos_theta *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * sin_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(cos_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0))) *
                    Kokkos::exp(-tanh_term) /
                    Kokkos::pow(
                        ((-Kokkos::pow(
                             (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) *
                                  (e * epsilon * r * sin_theta * cos_theta *
                                       xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                   e * sin_theta * xi / ((2.0 - tmp1))) -
                              sin_theta * cos_theta /
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0)),
                             2.0)) +
                         (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * cos_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                             (Kokkos::pow(
                                  (e * epsilon * r * sin_theta * cos_theta *
                                       xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                   e * sin_theta * xi / ((2.0 - tmp1))),
                                  2.0) +
                              ipow(cos_theta, 2) /
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0))),
                        (3.0 / 2.0)) +
                27.0336 * ipow(r - 1.0, 6) *
                    (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * cos_theta * xi / ((2.0 - tmp1))) *
                         (e * epsilon * r * sin_theta * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          e * sin_theta * xi / ((2.0 - tmp1))) -
                     sin_theta * cos_theta /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    Kokkos::exp(-tanh_term) * Kokkos::sin(11.0 * theta) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) -
                49.5616 * ipow(r - 1.0, 6) *
                    (Kokkos::pow((e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                     ipow(cos_theta, 2) /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    Kokkos::exp(-tanh_term) * Kokkos::cos(11.0 * theta) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) -
                4.5056 * ipow(r - 1.0, 6) *
                    (2.0 * epsilon * r * sin_theta * ipow(cos_theta, 2) /
                         Kokkos::pow(
                             (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                             2.0) +
                     (e * epsilon * r * sin_theta * cos_theta * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * sin_theta * xi / ((2.0 - tmp1))) *
                         (2.0 * e * (epsilon * epsilon) * (r * r) *
                              ipow(sin_theta, 2) * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) *
                               Kokkos::pow(
                                   (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                    1.0),
                                   (3.0 / 2.0))) -
                          4.0 * e * (epsilon * epsilon) * (r * r) *
                              ipow(sin_theta, 2) * cos_theta * xi /
                              (ipow(2.0 - tmp1, 3) *
                               (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                1.0)) -
                          4.0 * e * epsilon * r * ipow(sin_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          2.0 * e * epsilon * r * ipow(cos_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          2.0 * e * cos_theta * xi / ((2.0 - tmp1))) -
                     2.0 * sin_theta * cos_theta /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    Kokkos::exp(-tanh_term) * Kokkos::sin(11.0 * theta) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(
                             ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta /
                                  (Kokkos::sqrt(1.0 - 1.0 / 4.0 *
                                                          (epsilon * epsilon)) *
                                   (2.0 - tmp1))),
                             2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) -
                4.5056 * ipow(r - 1.0, 6) *
                    (Kokkos::pow((e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                     ipow(cos_theta, 2) /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    (1.0 / 2.0 *
                         (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * cos_theta * xi / ((2.0 - tmp1))) *
                              (e * epsilon * r * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * sin_theta * xi / ((2.0 - tmp1))) -
                          sin_theta * cos_theta /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                         ((-4.0) * epsilon * r * ipow(sin_theta, 2) *
                              cos_theta /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          2.0 *
                              ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) *
                              (e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) -
                               2.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               2.0 * e * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * epsilon * r * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) +
                          2.0 *
                              (e * epsilon * r * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * sin_theta * xi / ((2.0 - tmp1))) *
                              ((-e) * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 3) * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) +
                               2.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 3) * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               3.0 * e * epsilon * r * sin_theta * cos_theta *
                                   xi / (ipow(2.0 - tmp1, 2) * tmp1) -
                               e * sin_theta * xi / ((2.0 - tmp1))) +
                          2.0 * ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0) -
                          2.0 * ipow(cos_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) -
                     1.0 / 2.0 *
                         (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * cos_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                         (2.0 * epsilon * r * sin_theta * ipow(cos_theta, 2) /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          (e * epsilon * r * sin_theta * cos_theta * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * sin_theta * xi / ((2.0 - tmp1))) *
                              (2.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) -
                               4.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               4.0 * e * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               2.0 * e * epsilon * r * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               2.0 * e * cos_theta * xi / ((2.0 - tmp1))) -
                          2.0 * sin_theta * cos_theta /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) -
                     1.0 / 2.0 *
                         (Kokkos::pow((e * epsilon * r * sin_theta * cos_theta *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * sin_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(cos_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                         (2.0 * epsilon * r * ipow(sin_theta, 3) /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * cos_theta * xi / ((2.0 - tmp1))) *
                              ((-2.0) * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 3) * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) +
                               4.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 3) * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               6.0 * e * epsilon * r * sin_theta * cos_theta *
                                   xi / (ipow(2.0 - tmp1, 2) * tmp1) -
                               2.0 * e * sin_theta * xi / ((2.0 - tmp1))) +
                          2.0 * sin_theta * cos_theta /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0))) *
                    Kokkos::exp(-tanh_term) * Kokkos::sin(11.0 * theta) /
                    Kokkos::pow(
                        ((-Kokkos::pow(
                             (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) *
                                  (e * epsilon * r * sin_theta * cos_theta *
                                       xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                   e * sin_theta * xi / ((2.0 - tmp1))) -
                              sin_theta * cos_theta /
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0)),
                             2.0)) +
                         (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * cos_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                             (Kokkos::pow(
                                  (e * epsilon * r * sin_theta * cos_theta *
                                       xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                   e * sin_theta * xi / ((2.0 - tmp1))),
                                  2.0) +
                              ipow(cos_theta, 2) /
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0))),
                        (3.0 / 2.0)) -
                (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                      (ipow(2.0 - tmp1, 2) * tmp1) +
                  e * cos_theta * xi / ((2.0 - tmp1))) *
                     (e * epsilon * r * sin_theta * cos_theta * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * sin_theta * xi / ((2.0 - tmp1))) -
                 sin_theta * cos_theta /
                     (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    ((-27.0336) * r * ipow(r - 1.0, 5) *
                         Kokkos::sin(11.0 * theta) -
                     27.0336 * ipow(r - 1.0, 6) * Kokkos::sin(11.0 * theta)) *
                    Kokkos::exp(-tanh_term) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) -
                (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                      (ipow(2.0 - tmp1, 2) * tmp1) +
                  e * cos_theta * xi / ((2.0 - tmp1))) *
                     (e * epsilon * r * sin_theta * cos_theta * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * sin_theta * xi / ((2.0 - tmp1))) -
                 sin_theta * cos_theta /
                     (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    (2.4576 * r * ipow(r - 1.0, 5) * Kokkos::cos(11.0 * theta) +
                     2.4576 * ipow(r - 1.0, 6) * Kokkos::cos(11.0 * theta)) *
                    (1.0 / 2.0 *
                         (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * cos_theta * xi / ((2.0 - tmp1))) *
                              (e * epsilon * r * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * sin_theta * xi / ((2.0 - tmp1))) -
                          sin_theta * cos_theta /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                         ((-4.0) * epsilon * r * ipow(sin_theta, 2) *
                              cos_theta /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          2.0 *
                              ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) *
                              (e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) -
                               2.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               2.0 * e * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * epsilon * r * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) +
                          2.0 *
                              (e * epsilon * r * sin_theta * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * sin_theta * xi / ((2.0 - tmp1))) *
                              ((-e) * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 3) * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) +
                               2.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 3) * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               3.0 * e * epsilon * r * sin_theta * cos_theta *
                                   xi / (ipow(2.0 - tmp1, 2) * tmp1) -
                               e * sin_theta * xi / ((2.0 - tmp1))) +
                          2.0 * ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0) -
                          2.0 * ipow(cos_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) -
                     1.0 / 2.0 *
                         (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * cos_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                         (2.0 * epsilon * r * sin_theta * ipow(cos_theta, 2) /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          (e * epsilon * r * sin_theta * cos_theta * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * sin_theta * xi / ((2.0 - tmp1))) *
                              (2.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) -
                               4.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 2) * cos_theta * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               4.0 * e * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               2.0 * e * epsilon * r * ipow(cos_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               2.0 * e * cos_theta * xi / ((2.0 - tmp1))) -
                          2.0 * sin_theta * cos_theta /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) -
                     1.0 / 2.0 *
                         (Kokkos::pow((e * epsilon * r * sin_theta * cos_theta *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * sin_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(cos_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                         (2.0 * epsilon * r * ipow(sin_theta, 3) /
                              Kokkos::pow(
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0),
                                  2.0) +
                          ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                               (ipow(2.0 - tmp1, 2) * tmp1) +
                           e * cos_theta * xi / ((2.0 - tmp1))) *
                              ((-2.0) * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 3) * xi /
                                   (ipow(2.0 - tmp1, 2) *
                                    Kokkos::pow(
                                        (epsilon *
                                             (epsilon + 2.0 * r * cos_theta) +
                                         1.0),
                                        (3.0 / 2.0))) +
                               4.0 * e * (epsilon * epsilon) * (r * r) *
                                   ipow(sin_theta, 3) * xi /
                                   (ipow(2.0 - tmp1, 3) *
                                    (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                     1.0)) -
                               6.0 * e * epsilon * r * sin_theta * cos_theta *
                                   xi / (ipow(2.0 - tmp1, 2) * tmp1) -
                               2.0 * e * sin_theta * xi / ((2.0 - tmp1))) +
                          2.0 * sin_theta * cos_theta /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0))) *
                    Kokkos::exp(-tanh_term) /
                    Kokkos::pow(
                        ((-Kokkos::pow(
                             (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                   (ipow(2.0 - tmp1, 2) * tmp1) +
                               e * cos_theta * xi / ((2.0 - tmp1))) *
                                  (e * epsilon * r * sin_theta * cos_theta *
                                       xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                   e * sin_theta * xi / ((2.0 - tmp1))) -
                              sin_theta * cos_theta /
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0)),
                             2.0)) +
                         (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                           xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                       e * cos_theta * xi / ((2.0 - tmp1))),
                                      2.0) +
                          ipow(sin_theta, 2) /
                              (epsilon * (epsilon + 2.0 * r * cos_theta) +
                               1.0)) *
                             (Kokkos::pow(
                                  (e * epsilon * r * sin_theta * cos_theta *
                                       xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                   e * sin_theta * xi / ((2.0 - tmp1))),
                                  2.0) +
                              ipow(cos_theta, 2) /
                                  (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                   1.0))),
                        (3.0 / 2.0)) +
                6.0 *
                    (2.4576 * r * ipow(r - 1.0, 5) * Kokkos::cos(11.0 * theta) +
                     2.4576 * ipow(r - 1.0, 6) * Kokkos::cos(11.0 * theta)) *
                    (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * cos_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                     ipow(sin_theta, 2) /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    Kokkos::exp(-tanh_term) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0))) -
                (2.4576 * r * ipow(r - 1.0, 5) * Kokkos::cos(11.0 * theta) +
                 2.4576 * ipow(r - 1.0, 6) * Kokkos::cos(11.0 * theta)) *
                    ((-2.0) * epsilon * r * ipow(sin_theta, 2) * cos_theta /
                         Kokkos::pow(
                             (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0),
                             2.0) +
                     ((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * cos_theta * xi / ((2.0 - tmp1))) *
                         (e * (epsilon * epsilon) * (r * r) *
                              ipow(sin_theta, 2) * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) *
                               Kokkos::pow(
                                   (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                    1.0),
                                   (3.0 / 2.0))) -
                          2.0 * e * (epsilon * epsilon) * (r * r) *
                              ipow(sin_theta, 2) * cos_theta * xi /
                              (ipow(2.0 - tmp1, 3) *
                               (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                1.0)) -
                          2.0 * e * epsilon * r * ipow(sin_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          e * epsilon * r * ipow(cos_theta, 2) * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) +
                          e * cos_theta * xi / ((2.0 - tmp1))) +
                     (e * epsilon * r * sin_theta * cos_theta * xi /
                          (ipow(2.0 - tmp1, 2) * tmp1) +
                      e * sin_theta * xi / ((2.0 - tmp1))) *
                         ((-e) * (epsilon * epsilon) * (r * r) *
                              ipow(sin_theta, 3) * xi /
                              (ipow(2.0 - tmp1, 2) *
                               Kokkos::pow(
                                   (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                    1.0),
                                   (3.0 / 2.0))) +
                          2.0 * e * (epsilon * epsilon) * (r * r) *
                              ipow(sin_theta, 3) * xi /
                              (ipow(2.0 - tmp1, 3) *
                               (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                1.0)) -
                          3.0 * e * epsilon * r * sin_theta * cos_theta * xi /
                              (ipow(2.0 - tmp1, 2) * tmp1) -
                          e * sin_theta * xi / ((2.0 - tmp1))) +
                     ipow(sin_theta, 2) /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0) -
                     ipow(cos_theta, 2) /
                         (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                    Kokkos::exp(-tanh_term) /
                    Kokkos::sqrt(
                        (-Kokkos::pow(
                            (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                  (ipow(2.0 - tmp1, 2) * tmp1) +
                              e * cos_theta * xi / ((2.0 - tmp1))) *
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))) -
                             sin_theta * cos_theta /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)),
                            2.0)) +
                        (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) *
                                          xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                      e * cos_theta * xi / ((2.0 - tmp1))),
                                     2.0) +
                         ipow(sin_theta, 2) /
                             (epsilon * (epsilon + 2.0 * r * cos_theta) +
                              1.0)) *
                            (Kokkos::pow(
                                 (e * epsilon * r * sin_theta * cos_theta * xi /
                                      (ipow(2.0 - tmp1, 2) * tmp1) +
                                  e * sin_theta * xi / ((2.0 - tmp1))),
                                 2.0) +
                             ipow(cos_theta, 2) /
                                 (epsilon * (epsilon + 2.0 * r * cos_theta) +
                                  1.0)))) /
               Kokkos::sqrt(
                   (-Kokkos::pow(
                       (((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                             (ipow(2.0 - tmp1, 2) * tmp1) +
                         e * cos_theta * xi / ((2.0 - tmp1))) *
                            (e * epsilon * r * sin_theta * cos_theta * xi /
                                 (ipow(2.0 - tmp1, 2) * tmp1) +
                             e * sin_theta * xi / ((2.0 - tmp1))) -
                        sin_theta * cos_theta /
                            (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)),
                       2.0)) +
                   (Kokkos::pow(((-e) * epsilon * r * ipow(sin_theta, 2) * xi /
                                     (ipow(2.0 - tmp1, 2) * tmp1) +
                                 e * cos_theta * xi / ((2.0 - tmp1))),
                                2.0) +
                    ipow(sin_theta, 2) /
                        (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)) *
                       (Kokkos::pow((e * epsilon * r * sin_theta * cos_theta *
                                         xi / (ipow(2.0 - tmp1, 2) * tmp1) +
                                     e * sin_theta * xi / ((2.0 - tmp1))),
                                    2.0) +
                        ipow(cos_theta, 2) /
                            (epsilon * (epsilon + 2.0 * r * cos_theta) + 1.0)));
}

template class ManufacturedRHS<
    CurvilinearSolution<CzarnyToCartesian<R, Theta, X, Y>>>;
