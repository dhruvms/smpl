////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Andrew Dornbush
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     1. Redistributions of source code must retain the above copyright notice
//        this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. Neither the name of the copyright holder nor the names of its
//        contributors may be used to endorse or promote products derived from
//        this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////

/// \author Andrew Dornbush

#ifndef sbpl_collision_collision_operations_h
#define sbpl_collision_collision_operations_h

#include <sbpl_arm_planner/occupancy_grid.h>
#include <sbpl_collision_checking/robot_collision_state.h>

namespace sbpl {
namespace collision {

bool CheckSphereCollision(
    const OccupancyGrid& grid,
    RobotCollisionState& state,
    double padding,
    const SphereIndex& sidx,
    double& dist);

inline
bool CheckSphereCollision(
    const OccupancyGrid& grid,
    RobotCollisionState& state,
    double padding,
    const SphereIndex& sidx,
    double& dist)
{
    state.updateSphereState(sidx);
    const CollisionSphereState& ss = state.sphereState(sidx);

    // NOTE: no need to check bounds since getDistance will return the maximum
    // value for invalid cells

    // check for collision with world
    double obs_dist = grid.getDistanceFromPoint(ss.pos.x(), ss.pos.y(), ss.pos.z());
    const double effective_radius =
            ss.model->radius + grid.getHalfResolution() + padding;

    dist = obs_dist;
    return obs_dist > effective_radius;
}

std::vector<SphereIndex> GatherSphereIndices(
    const RobotCollisionState& state, int gidx);

} // namespace collision
} // namespace sbpl

#endif
