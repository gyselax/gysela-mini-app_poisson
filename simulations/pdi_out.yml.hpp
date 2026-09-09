// SPDX-License-Identifier: MIT
#pragma once

constexpr char const* const PDI_CFG = R"PDI_CFG(
metadata:
  r_size: int
  theta_size: int
  l_inf_error: double
  solver: char
  init_time: int
  solver_time: int
  
  r_range_extents: {type: array,  subtype: int64, size: 1 }
  r_range:
    type: array
    subtype: double
    size: [   '$r_range_extents[0]']
  
  theta_range_extents: {type: array,  subtype: int64, size: 1 }
  theta_range:
    type: array
    subtype: double
    size: [   '$theta_range_extents[0]']

  x_coords_extents: {type: array,  subtype: int64, size: 2 }
  x_coords:
    type: array
    subtype: double
    size: [   '$x_coords_extents[0]', '$x_coords_extents[1]']

  y_coords_extents: {type: array,  subtype: int64, size: 2 }
  y_coords:
    type: array
    subtype: double
    size: [  '$y_coords_extents[0]', '$y_coords_extents[1]']


data:
  electrical_potential_extents: {type: array, subtype: int64, size: 2 }
  electrical_potential: 
    type: array
    subtype: double
    size: [ '$electrical_potential_extents[0]', '$electrical_potential_extents[1]' ]


plugins:
  set_value:

  decl_hdf5:
    - file: 'output/poisson_${solver}_${r_size}_${theta_size}.h5'
      on_event: [last_iteration]
      collision_policy: replace_and_warn
      write: [r_size, theta_size,solver,init_time,solver_time,l_inf_error, r_range,theta_range, x_coords, y_coords,electrical_potential]
  #trace: ~
)PDI_CFG";