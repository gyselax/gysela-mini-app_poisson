// SPDX-License-Identifier: MIT
#pragma once
#include "geometry_r_theta.hpp"

struct A
{
};
struct B
{
};

struct GridPhi : UniformGridBase<A>
{
};

struct GridB : UniformGridBase<B>
{
};

using IdxRangeBatch = IdxRange<GridPhi>;
using IdxBatch = Idx<GridPhi>;
using IdxRangeBatched = IdxRange<GridPhi, GridR, GridTheta>;
using IdxBatched = Idx<GridPhi, GridR, GridTheta>;

using DFieldMemBatched = DFieldMem<IdxRangeBatched>;
using DFieldBatched = DField<IdxRangeBatched>;