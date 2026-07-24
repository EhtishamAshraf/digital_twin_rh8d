/*
                                UR5e Pick-and-Place Demo with RH8D Hand

This node demonstrates a complete pick-and-place sequence using MoveIt for the UR5e manipulator
and a custom hand service for RH8D Hand for grasping objects. The demo handles both a ball and a 
cylinder object, performing planning, execution, and basic error handling. 

Flow of the demo:
1. Open the hand before starting the pick sequence.
2. Ball Manipulation Sequence:
   a. Move to the ball pose.
   b. Close the hand to grasp the ball.
   c. Perform Cartesian motions to lift, move sideways, and shift forward.
   d. Open the hand to release the ball.
   e. Move to the cylinder pose.
3. Cylinder Manipulation Sequence:
   a. Move to the cylinder pose.
   b. Push the cylinder using Cartesian motion.
   c. Return to the home pose.
4. All motions are planned with MoveIt, and the hand is controlled asynchronously via a ROS2 service.
5. The node performs basic error handling for motion planning failures, execution failures, and 
   hand service timeouts.


Note:
A commented version of the file is avaialble here: /v4_rh8d_ws/src/ur5e_mujoco_control/src/cartesian_pickNplace.cpp
*/

#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

#include <rclcpp/rclcpp.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <rh8d_mujoco_interfaces/srv/hand_command.hpp>

using moveit::planning_interface::MoveGroupInterface;
using moveit::planning_interface::PlanningSceneInterface;
using moveit::core::MoveItErrorCode;
using HandCommand = rh8d_mujoco_interfaces::srv::HandCommand;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("ur5e_mujoco_control");

class UR5ePickDemo : public rclcpp::Node
{
public:
  UR5ePickDemo()
  : Node("ur5e_mujoco_control",
         rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
  {
    hand_client_ = this->create_client<HandCommand>("/rh8d/hand_command");
  }

  void run()
  {
    static const std::string PLANNING_GROUP = "ur5e_manipulator";

    move_group_ = std::make_shared<MoveGroupInterface>(shared_from_this(), PLANNING_GROUP);

    move_group_->setPlanningTime(10.0);
    move_group_->setNumPlanningAttempts(10);
    move_group_->setMaxVelocityScalingFactor(0.10);
    move_group_->setMaxAccelerationScalingFactor(0.10);

    move_group_->setPlanningPipelineId("ompl");
    move_group_->setPlannerId("RRTConnectkConfigDefault");

    RCLCPP_INFO(LOGGER, "Planning frame: %s", move_group_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End effector link: %s", move_group_->getEndEffectorLink().c_str());

    bool ball_sequence_ok = false;

    add_table_collision();
    rclcpp::sleep_for(std::chrono::seconds(1));

    auto ball_pose = make_pose(
        0.499, 0.506, -0.503, 0.492,
        0.8038, -0.099, 0.84);

    auto cylinder_pose = make_pose(
        0.025, -0.689, 0.724, -0.036,
        0.423, -0.111, 0.820);


    RCLCPP_INFO(LOGGER, "Opening hand before pick...");
    if (!call_hand_service("object", "open")) {
        RCLCPP_ERROR(LOGGER, "Failed to open hand. Check if hand service is available...");
        return;
    }


    // Ball Manipulation Sequence

    rclcpp::sleep_for(std::chrono::seconds(1));
    RCLCPP_INFO(LOGGER, "Going to ball pose...");
    if (!move_to_pose(ball_pose, "ball_pose")) {
        RCLCPP_ERROR(LOGGER, "***Failed to reach ball_pose. Going to cylinder pose...***");
        if (!move_to_pose(cylinder_pose, "cylinder_pose")) {
                RCLCPP_ERROR(LOGGER, "Failed to reach cylinder_pose as well. Exiting...");
                return;
        }
    } else {
        RCLCPP_INFO(LOGGER, "Closing hand with grasp service...");
        rclcpp::sleep_for(std::chrono::milliseconds(500));
        if (!call_hand_service("object", "grasp")) {
            RCLCPP_ERROR(LOGGER, "Failed to grasp object");
            return;
        } else {
              ball_sequence_ok = true;
              RCLCPP_INFO(LOGGER, "Object grasped successfully.");}
    }

    if (ball_sequence_ok) 
    {
      auto current_ball_position = move_group_->getCurrentPose().pose;
      RCLCPP_INFO(LOGGER, "***Current pose - ball picking: x=%.3f y=%.3f z=%.3f***",
                  current_ball_position.position.x, current_ball_position.position.y, current_ball_position.position.z);


      rclcpp::sleep_for(std::chrono::seconds(1));
      RCLCPP_INFO(LOGGER, "***Cartesian motion in Z...***");
      if (!move_cartesian_delta_locked_z(0.0, 0.0, 0.35, "Move Z", false)) {
          RCLCPP_ERROR(LOGGER, "***Failed: Cartesian move in Z***");
      }
      rclcpp::sleep_for(std::chrono::seconds(1));
      RCLCPP_INFO(LOGGER, "***Cartesian motion in Y...***");
      if (!move_cartesian_delta_locked_z(0.0, -0.1075, 0.0, "Move Y", true)) {
          RCLCPP_ERROR(LOGGER, "***Failed: Cartesian move in Y***");
      }
      rclcpp::sleep_for(std::chrono::seconds(1));
      RCLCPP_INFO(LOGGER, "***Cartesian motion in X...***");
      if (!move_cartesian_delta_locked_z(-0.235, 0.0, 0.0, "Move X", true)) {
          RCLCPP_ERROR(LOGGER, "***Failed: Cartesian move in X***");
      }

      rclcpp::sleep_for(std::chrono::seconds(1));
      RCLCPP_INFO(LOGGER, "***Opening hand to release object...***");
      rclcpp::sleep_for(std::chrono::milliseconds(500));
      if (!call_hand_service("object", "open")) {
          RCLCPP_ERROR(LOGGER, "***Failed to open hand for release***");
          return;
      }

      rclcpp::sleep_for(std::chrono::seconds(5));
      RCLCPP_INFO(LOGGER, "***Going to cylinder pose...***");
      if (!move_to_pose(cylinder_pose, "cylinder_pose")) {
          RCLCPP_ERROR(LOGGER, "***Failed to reach cylinder_pose. Returning to Home...***");
          move_to_named_target("Home");
          return;
      }
    }


    // Cylinder Manipulation Sequence

    auto current_cylinder_position = move_group_->getCurrentPose().pose;
    RCLCPP_INFO(LOGGER, "***Current pose - cylinder pushing: x=%.3f y=%.3f z=%.3f***",
                current_cylinder_position.position.x, current_cylinder_position.position.y, current_cylinder_position.position.z);

    rclcpp::sleep_for(std::chrono::seconds(1));
    RCLCPP_INFO(LOGGER, "***Pushing cylinder with Cartesian motion...***");
    if (!move_cartesian_delta_locked_z(-0.22, 0.0, 0.0, "Push Cylinder", true)) {
        RCLCPP_ERROR(LOGGER, "***Failed to push cylinder...***");
    }

    rclcpp::sleep_for(std::chrono::seconds(1));
    RCLCPP_INFO(LOGGER, "***Returning to Home...***");
    if (!move_to_named_target("Home")) {
        RCLCPP_ERROR(LOGGER, "***Failed to move arm to Home pose...Trying an alternative approach...***");
        rclcpp::sleep_for(std::chrono::seconds(1));
        RCLCPP_INFO(LOGGER, "***Cartesian motion in X...***");
        if (!move_cartesian_delta_locked_z(0.1, 0.0, 0.0, "Move X", true)) {
            RCLCPP_ERROR(LOGGER, "***Recovery move in +X also failed....***");
        }
        rclcpp::sleep_for(std::chrono::seconds(1));
        RCLCPP_INFO(LOGGER, "***Trying Home again...***");
        if (!move_to_named_target("Home")) {
            RCLCPP_ERROR(LOGGER, "***Second attempt to move Home also failed.***");
        }
        return;
    }


    rclcpp::sleep_for(std::chrono::seconds(1));
    RCLCPP_INFO(LOGGER, "***Pick and place sequence completed.***");
  }

private:
  geometry_msgs::msg::Pose make_pose(
    double ox, double oy, double oz, double ow,
    double px, double py, double pz)
  {
    geometry_msgs::msg::Pose pose;
    pose.orientation.x = ox;
    pose.orientation.y = oy;
    pose.orientation.z = oz;
    pose.orientation.w = ow;
    pose.position.x = px;
    pose.position.y = py;
    pose.position.z = pz;
    return pose;
  }


  bool move_to_named_target(const std::string &target_name)
  {
    move_group_->setStartStateToCurrentState();
    move_group_->setNamedTarget(target_name);

    MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_->plan(plan));

    if (!success) {
      RCLCPP_ERROR(LOGGER, "Planning failed named target: %s", target_name.c_str());
      return false;
    }

    RCLCPP_INFO(LOGGER, "Plan successful for %s, executing...", target_name.c_str());
    auto result = move_group_->execute(plan);

    move_group_->clearPoseTargets();

    if (result != MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "Execution failed for %s joint pose", target_name.c_str());
      return false;
    }

    RCLCPP_INFO(LOGGER, "Arm reached %s joint pose successfully.", target_name.c_str());
    return true;
  }




  bool move_to_pose(const geometry_msgs::msg::Pose &target_pose, const std::string &pose_name)
  {
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(target_pose);

    MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_->plan(plan));

    if (!success) {
      RCLCPP_ERROR(LOGGER, "Planning failed for %s", pose_name.c_str());
      move_group_->clearPoseTargets();
      return false;
    }

    RCLCPP_INFO(LOGGER, "Plan successful for %s, executing...", pose_name.c_str());
    auto result = move_group_->execute(plan);

    move_group_->clearPoseTargets();

    if (result != MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "Execution failed for %s", pose_name.c_str());
      return false;
    }

    RCLCPP_INFO(LOGGER, "%s reached successfully.", pose_name.c_str());
    return true;
  }

  bool call_hand_service(const std::string &mode, const std::string &gesture_name)
  {
    if (!hand_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(LOGGER, "Hand service /rh8d/hand_command not available");
      return false;
    }

    auto request = std::make_shared<HandCommand::Request>();
    request->mode = mode;
    request->gesture_name = gesture_name;

    auto future = hand_client_->async_send_request(request);

    auto status = future.wait_for(std::chrono::seconds(40));
    if (status != std::future_status::ready) {
      RCLCPP_ERROR(LOGGER, "Timed out waiting for hand service response");
      return false;
    }

    auto response = future.get();
    RCLCPP_INFO(LOGGER, "Hand service response: success=%s, message=%s",
                response->success ? "true" : "false",
                response->message.c_str());

    return response->success;
  }


bool move_cartesian_delta_locked_z(
  double x_delta, double y_delta, double z_delta,
  const std::string &pose_name,
  bool lock_z = true)
{
  rclcpp::sleep_for(std::chrono::milliseconds(300));
  move_group_->setStartStateToCurrentState();

  std::vector<geometry_msgs::msg::Pose> waypoints;

  geometry_msgs::msg::Pose start_pose = move_group_->getCurrentPose().pose;
  geometry_msgs::msg::Pose target_pose = start_pose;

  const double fixed_z = start_pose.position.z;

  target_pose.position.x += x_delta;
  target_pose.position.y += y_delta;
  target_pose.position.z += z_delta;

  if (lock_z) {
    target_pose.position.z = fixed_z;
  }

  RCLCPP_INFO(LOGGER, "Before: x=%.4f y=%.4f z=%.4f",
              start_pose.position.x, start_pose.position.y, start_pose.position.z);
  RCLCPP_INFO(LOGGER, "Target: x=%.4f y=%.4f z=%.4f",
              target_pose.position.x, target_pose.position.y, target_pose.position.z);

  // Only add target waypoint. MoveIt uses current state as start.
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory trajectory;

  const double eef_step = 0.002;
  const double jump_threshold = 0.0;

  double fraction = move_group_->computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory);

  RCLCPP_INFO(LOGGER, "Cartesian fraction for %s: %.3f", pose_name.c_str(), fraction);

  if (fraction < 0.95) {
    RCLCPP_ERROR(
      LOGGER,
      "Cartesian planning failed for %s. Fraction achieved: %.3f",
      pose_name.c_str(), fraction);
    return false;
  }

  MoveGroupInterface::Plan cartesian_plan;
  cartesian_plan.trajectory_ = trajectory;

  RCLCPP_INFO(LOGGER, "Cartesian path successful for %s, executing...", pose_name.c_str());
  auto result = move_group_->execute(cartesian_plan);

  if (result != MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(LOGGER, "Cartesian execution failed for %s", pose_name.c_str());
    return false;
  }

  rclcpp::sleep_for(std::chrono::milliseconds(300));
  auto end_pose = move_group_->getCurrentPose().pose;

  RCLCPP_INFO(LOGGER, "After : x=%.4f y=%.4f z=%.4f",
              end_pose.position.x, end_pose.position.y, end_pose.position.z);

  RCLCPP_INFO(LOGGER, "%s reached successfully with Cartesian motion.", pose_name.c_str());
  return true;
}








  void add_table_collision()
  {
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    const std::string frame_id = move_group_->getPlanningFrame();

    auto make_box = [&](const std::string& id,
                        double x, double y, double z,
                        double px, double py, double pz)
    {
      moveit_msgs::msg::CollisionObject obj;
      obj.id = id;
      obj.header.frame_id = frame_id;

      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {x, y, z};

      geometry_msgs::msg::Pose pose;
      pose.orientation.w = 1.0;
      pose.position.x = px;
      pose.position.y = py;
      pose.position.z = pz;

      obj.primitives.push_back(primitive);
      obj.primitive_poses.push_back(pose);
      obj.operation = moveit_msgs::msg::CollisionObject::ADD;

      return obj;
    };

    // MuJoCo body origin: pos="0 0 0.73"
    // MoveIt uses full box dimensions, MuJoCo uses half-size

    collision_objects.push_back(
      make_box("stand_base_col",
               2.0, 0.72, 0.40,
               0.1, -0.137, 0.5301));

    collision_objects.push_back(
      make_box("stand_column_col",
               0.10, 0.18, 0.80,
               0.0, 0.135, 1.045));

    collision_objects.push_back(
      make_box("stand_top_col",
               0.48, 0.182, 0.052,
               0.0, 0.135, 1.42));

    collision_objects.push_back(
      make_box("stand_back_col",
               2.0, 0.005, 0.60,
               0.1, 0.227, 1.03));

    collision_objects.push_back(
      make_box("stand_right_side_col",
               0.005, 0.725, 0.60,
               1.1, -0.135, 1.03));

    planning_scene_interface_.applyCollisionObjects(collision_objects);
    RCLCPP_INFO(LOGGER, "Added table collision objects to MoveIt");
  }

  std::shared_ptr<MoveGroupInterface> move_group_;
  PlanningSceneInterface planning_scene_interface_;
  rclcpp::Client<HandCommand>::SharedPtr hand_client_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<UR5ePickDemo>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  std::thread spinner([&executor]() { executor.spin(); });

  rclcpp::sleep_for(std::chrono::seconds(2));
  node->run();

  rclcpp::shutdown();
  spinner.join();
  return 0;
}