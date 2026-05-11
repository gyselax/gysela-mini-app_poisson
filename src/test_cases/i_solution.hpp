// SPDX-License-Identifier: MIT
#pragma once
#include <ddc/ddc.hpp>

#include <coord_transformation_tools.hpp>
#include <ddc_aliases.hpp>
#include <geometry_r_theta.hpp>

namespace concepts {

/**
 * @brief Concept for the exact solutions of the Poisson-like equation.
 *
 * @see PolarSplineFEMPoissonLikeSolver
 * @see VlasovPoissonSolver
 */
		template <typename T>
		concept Solution = requires(T const& t, Coord<R, Theta> const& coord) {
				{t(coord)
		} -> std::same_as<double>;
		};

};
