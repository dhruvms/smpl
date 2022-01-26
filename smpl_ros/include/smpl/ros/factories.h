#ifndef SMPL_ROS_FACTORIES_H
#define SMPL_ROS_FACTORIES_H

// standard includes
#include <memory>
#include <string>
#include <vector>
#include <sbpl/heuristics/heuristic.h>

class SBPLPlanner;

namespace smpl {

class CollisionChecker;
class OccupancyGrid;
class PlanningParams;
class RobotPlanningSpace;
class RobotModel;
class RobotHeuristic;

bool IsMultiDOFJointVariable(
    const std::string& name,
    std::string* joint_name = NULL,
    std::string* local_name = NULL);

//////////////////////////////
// Planning Space Factories //
//////////////////////////////

auto MakeManipLattice(
    RobotModel* robot,
    CollisionChecker* checker,
    const PlanningParams& params,
    const OccupancyGrid* grid)
    -> std::unique_ptr<RobotPlanningSpace>;

auto MakeManipLatticeEGraph(
    RobotModel* robot,
    CollisionChecker* checker,
    const PlanningParams& params,
    const OccupancyGrid* grid)
    -> std::unique_ptr<RobotPlanningSpace>;

auto MakeWorkspaceLattice(
    RobotModel* robot,
    CollisionChecker* checker,
    const PlanningParams& params,
    const OccupancyGrid* grid)
    -> std::unique_ptr<RobotPlanningSpace>;

auto MakeWorkspaceLatticeEGraph(
    RobotModel* robot,
    CollisionChecker* checker,
    const PlanningParams& params,
    const OccupancyGrid* grid)
    -> std::unique_ptr<RobotPlanningSpace>;

auto MakeAdaptiveWorkspaceLattice(
    RobotModel* robot,
    CollisionChecker* checker,
    const PlanningParams& params,
    const OccupancyGrid* grid)
    -> std::unique_ptr<RobotPlanningSpace>;

/////////////////////////
// Heuristic Factories //
/////////////////////////

auto MakeMultiFrameBFSHeuristic(
    RobotPlanningSpace* space,
    const PlanningParams& params,
    const OccupancyGrid* grid)
    -> std::unique_ptr<RobotHeuristic>;

auto MakeBFSHeuristic(
    RobotPlanningSpace* space,
    const PlanningParams& param,
    const OccupancyGrid* grid)
    -> std::unique_ptr<RobotHeuristic>;

auto MakeEuclidDistHeuristic(
    RobotPlanningSpace* space,
    const PlanningParams& params)
    -> std::unique_ptr<RobotHeuristic>;

auto MakeJointDistHeuristic(
    RobotPlanningSpace* space,
    const PlanningParams& params)
    -> std::unique_ptr<RobotHeuristic>;

auto MakeDijkstraEgraphHeuristic3D(
    RobotPlanningSpace* space,
    const PlanningParams& params,
    const OccupancyGrid* grid)
    -> std::unique_ptr<RobotHeuristic>;

auto MakeJointDistEGraphHeuristic(
    RobotPlanningSpace* space,
    const PlanningParams& params)
    -> std::unique_ptr<RobotHeuristic>;

//////////////////////
// Search Factories //
//////////////////////

auto MakeARAStar(
    RobotPlanningSpace* space,
    std::vector<Heuristic*>& heuristic,
    const PlanningParams& params)
    -> std::unique_ptr<SBPLPlanner>;

auto MakeAWAStar(
    RobotPlanningSpace* space,
    std::vector<Heuristic*>& heuristic,
    const PlanningParams& params)
    -> std::unique_ptr<SBPLPlanner>;

auto MakeMHAStar(
    RobotPlanningSpace* space,
    std::vector<Heuristic*>& heuristics,
    const PlanningParams& params)
    -> std::unique_ptr<SBPLPlanner>;

auto MakeLARAStar(
    RobotPlanningSpace* space,
    std::vector<Heuristic*>& heuristic,
    const PlanningParams& params)
    -> std::unique_ptr<SBPLPlanner>;

// auto MakeEGWAStar(
//     RobotPlanningSpace* space,
//     std::vector<Heuristic*>& heuristic,
//     const PlanningParams& params)
//     -> std::unique_ptr<SBPLPlanner>;

// auto MakePADAStar(
//     RobotPlanningSpace* space,
//     std::vector<Heuristic*>& heuristic,
//     const PlanningParams& params)
//     -> std::unique_ptr<SBPLPlanner>;

} // namespace smpl

#endif

