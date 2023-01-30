#include <sbpl_kdl_robot_model/pushing_kdl_robot_model.h>

#include <sbpl_kdl_robot_model/kdl_robot_model.h>

// system includes
#include <eigen_conversions/eigen_kdl.h>
#include <kdl/frames.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <leatherman/print.h>
#include <leatherman/utils.h>
#include <ros/console.h>
#include <smpl/angles.h>
#include <smpl/time.h>
#include <smpl/stl/memory.h>

namespace smpl {

bool PushingKDLRobotModel::init(
	const std::string& robot_description,
	const std::string& base_link,
	const std::string& tip_link,
	int free_angle)
{
	if (!KDLRobotModel::init(
			robot_description, base_link, tip_link, free_angle))
	{
		return false;
	}

	m_cart_to_jnt_vel_solver = make_unique<KDL::ChainIkSolverVel_wdls>(m_chain);
	Eigen::Matrix<double, 6, 6> weights_ts = Eigen::Matrix<double, 6, 6>::Identity();
	weights_ts(3, 3) = 0;
	weights_ts(4, 4) = 0;
	weights_ts(5, 5) = 0;

	m_ik_solver_pos = make_unique<KDL::ChainIkSolverPos_LMA>(
								m_chain,
								weights_ts.diagonal());

	m_Jq_solver = make_unique<KDL::ChainJntToJacSolver>(m_chain);

	return true;
}

bool PushingKDLRobotModel::computeInverseVelocity(
		const RobotState& jnt_positions,
		const std::vector<double>& cart_velocities,
		RobotState& jnt_velocities)
{
	KDL::JntArray  q_(jnt_positions.size());
	KDL::JntArray  qdot_(jnt_positions.size());
	KDL::Twist     xdot_;
	jnt_velocities.resize(jnt_positions.size());

	for (size_t i = 0; i < jnt_positions.size(); ++i) {
		q_(i) = jnt_positions[i];
	}

	xdot_.vel(0) = cart_velocities[0];
	xdot_.vel(1) = cart_velocities[1];
	xdot_.vel(2) = cart_velocities[2];
	if(cart_velocities.size() == 3)
	{
	  xdot_.rot(0) = 0;
	  xdot_.rot(1) = 0;
	  xdot_.rot(2) = 0;
	}
	else
	{
	  xdot_.rot(0) = cart_velocities[3];
	  xdot_.rot(1) = cart_velocities[4];
	  xdot_.rot(2) = cart_velocities[5];
	}

	if (!m_cart_to_jnt_vel_solver->CartToJnt(q_, xdot_, qdot_) < 0) {
		ROS_WARN("Failed to find inverse joint velocities");
		return false;
	}

	for (size_t i = 0; i < jnt_velocities.size(); ++i) {
		jnt_velocities[i] = qdot_(i);
	}
	return true;
}

bool PushingKDLRobotModel::computeJacobian(
	const RobotState& jnt_positions,
	KDL::Jacobian& Jq)
{
	KDL::JntArray  q_(jnt_positions.size());
	for (size_t i = 0; i < jnt_positions.size(); ++i) {
		q_(i) = jnt_positions[i];
	}

	Jq.resize(m_chain.getNrOfJoints());
	int error = m_Jq_solver->JntToJac(q_, Jq);
	if (error < 0) {
		return false;
	}

	return true;
}

bool PushingKDLRobotModel::computeJacobian(
	const RobotState& jnt_positions,
	Eigen::MatrixXd& Jq)
{
	KDL::Jacobian Jq_KDL;
	if (this->computeJacobian(jnt_positions, Jq_KDL))
	{
		Jq = Jq_KDL.data;
		return true;
	}

	return false;
}

bool PushingKDLRobotModel::computeIKPos(
	const Eigen::Affine3d& pose,
	const RobotState& start,
	RobotState& solution)
{
	// transform into kinematics and convert to kdl
	auto* T_map_kinematics = GetLinkTransform(&this->robot_state, m_kinematics_link);
	KDL::Frame frame_des;
	tf::transformEigenToKDL(T_map_kinematics->inverse() * pose, frame_des);

	// seed configuration
	for (size_t i = 0; i < start.size(); i++) {
		m_jnt_pos_in(i) = start[i];
	}

	// must be normalized for CartToJntSearch
	NormalizeAngles(this, &m_jnt_pos_in);

	auto initial_guess = m_jnt_pos_in(m_free_angle);

	auto start_time = smpl::clock::now();
	auto loop_time = 0.0;
	auto count = 0;

	auto num_positive_increments =
			(int)((GetSolverMinPosition(this, m_free_angle) - initial_guess) /
					this->m_search_discretization);
	auto num_negative_increments =
			(int)((initial_guess - GetSolverMinPosition(this, m_free_angle)) /
					this->m_search_discretization);

	while (loop_time < this->m_timeout) {
		if (m_ik_solver_pos->CartToJnt(m_jnt_pos_in, frame_des, m_jnt_pos_out) >= 0) {
			NormalizeAngles(this, &m_jnt_pos_out);
			solution.resize(start.size());
			for (size_t i = 0; i < solution.size(); ++i) {
				solution[i] = m_jnt_pos_out(i);
			}
			return true;
		}
		if (!getCount(count, num_positive_increments, -num_negative_increments)) {
			return false;
		}
		m_jnt_pos_in(m_free_angle) = initial_guess + this->m_search_discretization * count;
		ROS_DEBUG("%d, %f", count, m_jnt_pos_in(m_free_angle));
		loop_time = to_seconds(smpl::clock::now() - start_time);
	}

	if (loop_time >= this->m_timeout) {
		ROS_DEBUG("IK Timed out in %f seconds", this->m_timeout);
		return false;
	} else {
		ROS_DEBUG("No IK solution was found");
		return false;
	}
	return false;
}

auto PushingKDLRobotModel::getExtension(size_t class_code) -> Extension*
{
	if (class_code == GetClassCode<InverseKinematicsInterface>()
		|| class_code == GetClassCode<InverseVelocityInterface>()) return this;

	return URDFRobotModel::getExtension(class_code);
}

} // namespace smpl
