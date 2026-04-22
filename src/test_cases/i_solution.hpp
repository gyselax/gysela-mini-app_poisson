// SPDX-License-Identifier: MIT
#pragma once
#include <coord_transformation_tools.hpp>
#include <ddc/ddc.hpp>
#include <ddc_aliases.hpp>
#include <geometry_r_theta.hpp>

/**
 * @brief Base class for the exact solutions of the Poisson-like equation.
 *
 * @see PolarSplineFEMPoissonLikeSolver
 * @see VlasovPoissonSolver
 */
template <class CurvilinearToCartesian>
class ISolution {
   public:
    /**
     * @brief Type the mapping function which converts the logical (polar)
     * coordinates into the physical (Cartesian) coordinates.
     */
    using coordinate_converter_type = CurvilinearToCartesian;
    static_assert(is_coord_transform_with_o_point_v<CurvilinearToCartesian>);

   private:
    using X = typename CurvilinearToCartesian::cartesian_tag_x;
    using Y = typename CurvilinearToCartesian::cartesian_tag_y;
    using R = typename CurvilinearToCartesian::curvilinear_tag_r;
    using Theta = typename CurvilinearToCartesian::curvilinear_tag_theta;

   protected:
    /**
     * @brief The mapping function which converts the logical (polar)
     * coordinates into the physical (Cartesian) coordinates.
     */
    CurvilinearToCartesian const& m_coordinate_converter;

   public:
    virtual ~ISolution() = default;

    /**
     * @brief Instantiate an ISolution.
     *
     * @param[in] coordinate_converter
     *      The mapping function which converts the logical (polar)
     *      coordinates into the physical (Cartesian) coordinates.
     */
    explicit ISolution(CurvilinearToCartesian const& coordinate_converter)
        : m_coordinate_converter(coordinate_converter) {}
    /**
     * @brief Get the value of the function at a given coordinate.
     *
     * @param[in] coord
     *      The given polar coordinate.
     *
     * @return The value of the function at a given coordinate.
     */
    virtual double operator()(Coord<R, Theta> const& coord) const = 0;
};

