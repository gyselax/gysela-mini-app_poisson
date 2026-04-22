// SPDX-License-Identifier: MIT
#pragma once
#include <cmath>
#include <ddc/ddc.hpp>
#include <ddc_aliases.hpp>

#include "geometry_r_theta.hpp"

/**
 * @brief Defining the corresponding RHS of the Poisson equation
 * for a given exact solution.
 */
template <class ChosenSolution>
class ManufacturedRHS {
   public:
    /**
     * @brief Type the chosen solution of the Poisson equation.
     */
    using CurvilinearToCartesian =
        typename ChosenSolution::coordinate_converter_type;

   private:
    CurvilinearToCartesian m_coordinate_converter;

   public:
    /**
     * @brief Instantiate a ManufacturedRHS.
     *
     * @param[in] coordinate_converter
     *      The mapping function which converts the logical (polar)
     *      coordinates into the physical (Cartesian) coordinates.
     */
    explicit ManufacturedRHS(
        CurvilinearToCartesian const& coordinate_converter)
        : m_coordinate_converter(coordinate_converter) {}

    /**
     * @brief Get the value of the RHS at the O-point.
     *
     * @param[in] coord
     *      The given polar coordinate.
     *
     * @return The value of the RHS at the O-point.
     */
    KOKKOS_FUNCTION double solution_at_pole(Coord<R, Theta> const& coord) const;

    /**
     * @brief Get the value of the RHS on the domain excluding the O-point.
     *
     * @param[in] coord
     *      The given polar coordinate.
     *
     * @return The value of the RHS on the domain excluding the O-point.
     */
    KOKKOS_FUNCTION double non_singular_solution(
        Coord<R, Theta> const& coord) const;

    /**
     * @brief Get the value of the RHS at any point of the domain.
     *
     * @param[in] coord
     *      The given polar coordinate.
     *
     * @return The value of the RHS at any point of the domain.
     */
    KOKKOS_FUNCTION double operator()(Coord<R, Theta> const& coord) const {
        if (ddc::get<R>(coord) == 0.0) {
            return solution_at_pole(coord);
        } else {
            return non_singular_solution(coord);
        }
    }
};
