/*
                                        Gestures Demo: UR5e and RH8D Hand
This node demonstrates using MoveIt to plan and execute a sequence of arm poses, while also calling a custom hand gesture service to change the robot's hand configuration.
Flow of the demo:
1. The arm moves to a "Gestures" pose.
2. The hand executes a series of gestures: fully open, fully closed, index point, middle point, victory sign, and then fully open again.
3. Finally, the arm returns to the home pose.

Note:
A commented version of the file is avaialble here: /v4_rh8d_ws/src/ur5e_mujoco_control/src/gestures.cpp
*/

#include <memory>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <map>

#include <rclcpp/rclcpp.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <rh8d_mujoco_interfaces/srv/hand_command.hpp>

using moveit::planning_interface::MoveGroupInterface;
using moveit::planning_interface::PlanningSceneInterface;
using moveit::core::MoveItErrorCode;
using HandCommand = rh8d_mujoco_interfaces::srv::HandCommand;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("gestures");

class UR5eGestureDemo : public rclcpp::Node
{
public:
  UR5eGestureDemo()
  : Node("gestures",
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

    add_table_collision();
    rclcpp::sleep_for(std::chrono::seconds(1));

    auto names = move_group_->getNamedTargets();
    for (const auto &name : names) {
      RCLCPP_INFO(LOGGER, "Named target available: %s", name.c_str());
    }

    if (!move_to_named_target("Gestures")) {
      RCLCPP_ERROR(LOGGER, "Failed to move arm to Gestures pose");
      return;
    }

    rclcpp::sleep_for(std::chrono::seconds(1));

    std::vector<std::string> gesture_names = {
      "fully_open",
      "fully_closed",
      "index_point",
      "middle_point",
      "victory_sign",
      "fully_open"
    };

    for (const auto &gesture : gesture_names) {
      RCLCPP_INFO(LOGGER, "Calling hand gesture: %s", gesture.c_str());

      if (!call_hand_service("gesture", gesture)) {
        RCLCPP_WARN(LOGGER, "Gesture failed: %s", gesture.c_str());
      }

      rclcpp::sleep_for(std::chrono::seconds(3));
    }

    RCLCPP_INFO(LOGGER, "Gesture sequence completed... Moving arm back to home pose");

    if (!move_to_named_target("Home")) {
      RCLCPP_ERROR(LOGGER, "Failed to move arm to home pose");
      return;
    }
  }

private:
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

    auto status = future.wait_for(std::chrono::seconds(20));
    if (status != std::future_status::ready) {
      RCLCPP_ERROR(LOGGER, "Timed out waiting for hand service response");
      return false;
    }

    auto response = future.get();
    RCLCPP_INFO(LOGGER, "Hand response: success=%s, message=%s",
                response->success ? "true" : "false",
                response->message.c_str());

    return response->success;
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

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<UR5eGestureDemo>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  std::thread spinner([&executor]() { executor.spin(); });

  rclcpp::sleep_for(std::chrono::seconds(2));
  node->run();

  rclcpp::shutdown();
  spinner.join();
  return 0;
}