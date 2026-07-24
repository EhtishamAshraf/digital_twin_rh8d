// Copyright (c) 2025 Sangtaek Lee
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "mujoco_ros2_control/mujoco_system.hpp"

namespace mujoco_ros2_control
{
MujocoSystem::MujocoSystem() : logger_(rclcpp::get_logger("")) {}

std::vector<hardware_interface::StateInterface> MujocoSystem::export_state_interfaces()
{
  return std::move(state_interfaces_);
}

std::vector<hardware_interface::CommandInterface> MujocoSystem::export_command_interfaces()
{
  return std::move(command_interfaces_);
}

hardware_interface::return_type MujocoSystem::read(
  const rclcpp::Time & /* time */, const rclcpp::Duration & /* period */)
{
  // Command-space joint / actuator states
  for (auto &joint_state : joint_states_)
  {
    if (joint_state.is_mujoco_actuator)
    {
      if (joint_state.has_mujoco_joint_state)
      {
        joint_state.position = mj_data_->qpos[joint_state.mj_pos_adr];
        joint_state.velocity = mj_data_->qvel[joint_state.mj_vel_adr];
        joint_state.effort = mj_data_->qfrc_actuator[joint_state.mj_vel_adr];
        continue;
      }

      // For actuator-backed interfaces:
      // position -> current ctrl value, only when no matching MuJoCo joint exists
      // effort   -> actuator force
      joint_state.position = mj_data_->ctrl[joint_state.mj_actuator_id];
      joint_state.velocity = 0.0;
      joint_state.effort = mj_data_->actuator_force[joint_state.mj_actuator_id];
      continue;
    }

    // Regular MuJoCo joint-backed interfaces
    joint_state.position = mj_data_->qpos[joint_state.mj_pos_adr];
    joint_state.velocity = mj_data_->qvel[joint_state.mj_vel_adr];
    joint_state.effort = mj_data_->qfrc_applied[joint_state.mj_vel_adr];
  }

  //########## Added support for "URDF joint state export for tendon-driven MuJoCo models" during this project ##########
  for (auto &joint_state : urdf_joint_states_)
  {
    joint_state.position = mj_data_->qpos[joint_state.mj_pos_adr];
    joint_state.velocity = mj_data_->qvel[joint_state.mj_vel_adr];
    joint_state.effort = mj_data_->qfrc_actuator[joint_state.mj_vel_adr];
  }
  //########## Added support for "URDF joint state export for tendon-driven MuJoCo models" during this project ##########

  // IMU Sensor data
  for (auto &data : imu_sensor_data_)
  {
    data.orientation.data.w() = mj_data_->sensordata[data.orientation.mj_sensor_index];
    data.orientation.data.x() = mj_data_->sensordata[data.orientation.mj_sensor_index + 1];
    data.orientation.data.y() = mj_data_->sensordata[data.orientation.mj_sensor_index + 2];
    data.orientation.data.z() = mj_data_->sensordata[data.orientation.mj_sensor_index + 3];

    data.angular_velocity.data.x() = mj_data_->sensordata[data.angular_velocity.mj_sensor_index];
    data.angular_velocity.data.y() =
      mj_data_->sensordata[data.angular_velocity.mj_sensor_index + 1];
    data.angular_velocity.data.z() =
      mj_data_->sensordata[data.angular_velocity.mj_sensor_index + 2];

    data.linear_acceleration.data.x() =
      mj_data_->sensordata[data.linear_acceleration.mj_sensor_index];
    data.linear_acceleration.data.y() =
      mj_data_->sensordata[data.linear_acceleration.mj_sensor_index + 1];
    data.linear_acceleration.data.z() =
      mj_data_->sensordata[data.linear_acceleration.mj_sensor_index + 2];
  }

  // FT Sensor data
  for (auto &data : ft_sensor_data_)
  {
    data.force.data.x() = -mj_data_->sensordata[data.force.mj_sensor_index];
    data.force.data.y() = -mj_data_->sensordata[data.force.mj_sensor_index + 1];
    data.force.data.z() = -mj_data_->sensordata[data.force.mj_sensor_index + 2];

    data.torque.data.x() = -mj_data_->sensordata[data.torque.mj_sensor_index];
    data.torque.data.y() = -mj_data_->sensordata[data.torque.mj_sensor_index + 1];
    data.torque.data.z() = -mj_data_->sensordata[data.torque.mj_sensor_index + 2];
  }

  //########## Added support for "Force Tip sensor and Palm Distance sensor" during this project ##########
  for (auto &data : scalar_sensor_data_)
  {
    data.value = mj_data_->sensordata[data.mj_sensor_index];
  }

  for (auto &data : vec3_sensor_data_)
  {
    data.data.x() = mj_data_->sensordata[data.mj_sensor_index];
    data.data.y() = mj_data_->sensordata[data.mj_sensor_index + 1];
    data.data.z() = mj_data_->sensordata[data.mj_sensor_index + 2];
  }
  //########## Added support for "Force Tip sensor and Palm Distance sensor" during this project ##########


  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MujocoSystem::write(
  const rclcpp::Time & /* time */, const rclcpp::Duration &period)
{
  // Update mimic "logical" entries first
  for (auto &joint_state : joint_states_)
  {
    if (joint_state.is_mimic)
    {
      joint_state.position_command =
        joint_state.mimic_multiplier *
        joint_states_.at(joint_state.mimicked_joint_index).position_command;
      joint_state.velocity_command =
        joint_state.mimic_multiplier *
        joint_states_.at(joint_state.mimicked_joint_index).velocity_command;
      joint_state.effort_command =
        joint_state.mimic_multiplier *
        joint_states_.at(joint_state.mimicked_joint_index).effort_command;
    }
  }

  // Write commands
  for (auto &joint_state : joint_states_)
  {
    if (joint_state.is_mujoco_actuator)
    {
      if (joint_state.is_position_control_enabled)
      {
        mj_data_->ctrl[joint_state.mj_actuator_id] = clamp(
          joint_state.position_command,
          joint_state.min_position_command,
          joint_state.max_position_command);
      }
      else if (joint_state.is_effort_control_enabled)
      {
        mj_data_->ctrl[joint_state.mj_actuator_id] = clamp(
          joint_state.effort_command,
          joint_state.min_effort_command,
          joint_state.max_effort_command);
      }
      else if (joint_state.is_velocity_control_enabled)
      {
        mj_data_->ctrl[joint_state.mj_actuator_id] = clamp(
          joint_state.velocity_command,
          joint_state.min_velocity_command,
          joint_state.max_velocity_command);
      }

      continue;
    }

    // Original joint-backed behavior
    if (joint_state.is_position_control_enabled)
    {
      if (joint_state.is_pid_enabled)
      {
        double error = joint_state.position_command - mj_data_->qpos[joint_state.mj_pos_adr];
        mj_data_->qfrc_applied[joint_state.mj_vel_adr] =
          joint_state.position_pid.computeCommand(error, period.nanoseconds());
      }
      else
      {
        mj_data_->qpos[joint_state.mj_pos_adr] = joint_state.position_command;
      }
    }

    if (joint_state.is_velocity_control_enabled)
    {
      if (joint_state.is_pid_enabled)
      {
        double error = joint_state.velocity_command - mj_data_->qvel[joint_state.mj_vel_adr];
        mj_data_->qfrc_applied[joint_state.mj_vel_adr] =
          joint_state.velocity_pid.computeCommand(error, period.nanoseconds());
      }
      else
      {
        mj_data_->qvel[joint_state.mj_vel_adr] = joint_state.velocity_command;
      }
    }

    if (joint_state.is_effort_control_enabled)
    {
      double min_eff, max_eff;
      min_eff = joint_state.joint_limits.has_effort_limits
                  ? -1 * joint_state.joint_limits.max_effort
                  : std::numeric_limits<double>::lowest();
      min_eff = std::max(min_eff, joint_state.min_effort_command);

      max_eff = joint_state.joint_limits.has_effort_limits
                  ? joint_state.joint_limits.max_effort
                  : std::numeric_limits<double>::max();
      max_eff = std::min(max_eff, joint_state.max_effort_command);

      mj_data_->qfrc_applied[joint_state.mj_vel_adr] =
        clamp(joint_state.effort_command, min_eff, max_eff);
    }
  }

  return hardware_interface::return_type::OK;
}

bool MujocoSystem::init_sim(
  mjModel *mujoco_model, mjData *mujoco_data, const urdf::Model &urdf_model,
  const hardware_interface::HardwareInfo &hardware_info)
{
  mj_model_ = mujoco_model;
  mj_data_ = mujoco_data;

  logger_ = rclcpp::get_logger("mujoco_system");

  ft_sensor_data_.clear();
  imu_sensor_data_.clear();
  vec3_sensor_data_.clear();
  scalar_sensor_data_.clear();

  ft_sensor_data_.reserve(hardware_info.sensors.size());
  imu_sensor_data_.reserve(hardware_info.sensors.size());
  vec3_sensor_data_.reserve(hardware_info.sensors.size());
  scalar_sensor_data_.reserve(hardware_info.sensors.size());

  register_joints(urdf_model, hardware_info);
  register_urdf_joint_states(urdf_model);
  register_sensors(urdf_model, hardware_info);

  set_initial_pose();
  return true;
}

//#################################################################################################
// Added support for "explicit ros2_control-to-MuJoCo actuator name mapping".
//
// Purpose:
//   In some cases, the name exposed to ros2_control should be different from
//   the actual MuJoCo actuator name.
//
//   Example:
//     ros2_control joint name: forearm:1--base:1
//     MuJoCo actuator name:    pos_forearm
//
//   If the ros2_control joint contains a parameter called "mujoco_actuator",
//   this plugin uses that parameter value as the MuJoCo actuator name and
//   directly registers the interface as actuator-backed.
//
//   This allows cleaner ROS 2 control interface names while still commanding
//   the correct MuJoCo actuator internally.
//#################################################################################################
void MujocoSystem::register_joints(
  const urdf::Model &urdf_model, const hardware_interface::HardwareInfo &hardware_info)
{
  joint_states_.resize(hardware_info.joints.size());

  for (size_t joint_index = 0; joint_index < hardware_info.joints.size(); joint_index++)
  {
    auto joint = hardware_info.joints.at(joint_index);

    // By default, use the ros2_control joint name for MuJoCo lookup.
    // But if a mapping param is provided, use that instead.
    std::string mujoco_name = joint.name;
    bool prefer_actuator = false;

    auto actuator_param_it = joint.parameters.find("mujoco_actuator");
    if (actuator_param_it != joint.parameters.end())
    {
      mujoco_name = actuator_param_it->second;
      prefer_actuator = true;
    }

    int mujoco_joint_id = -1;
    int mujoco_actuator_id = -1;

    if (prefer_actuator)
    {
      mujoco_actuator_id = mj_name2id(mj_model_, mjOBJ_ACTUATOR, mujoco_name.c_str());
      if (mujoco_actuator_id == -1)
      {
        RCLCPP_ERROR_STREAM(
          logger_,
          "Failed to find mapped MuJoCo actuator, ros2_control joint name: "
            << joint.name << ", mapped actuator name: " << mujoco_name);
        continue;
      }

      mujoco_joint_id = mj_name2id(mj_model_, mjOBJ_JOINT, joint.name.c_str());
    }
    else
    {
      mujoco_joint_id = mj_name2id(mj_model_, mjOBJ_JOINT, mujoco_name.c_str());

      if (mujoco_joint_id == -1)
      {
        mujoco_actuator_id = mj_name2id(mj_model_, mjOBJ_ACTUATOR, mujoco_name.c_str());
        if (mujoco_actuator_id == -1)
        {
          RCLCPP_ERROR_STREAM(
            logger_,
            "Failed to find joint or actuator in mujoco model, ros2_control joint name: "
              << joint.name << ", lookup name: " << mujoco_name);
          continue;
        }
      }
    }

    JointState joint_state;
    joint_state.name = joint.name;

    if (mujoco_joint_id != -1)
    {
      joint_state.mj_joint_type = mj_model_->jnt_type[mujoco_joint_id];
      joint_state.mj_pos_adr = mj_model_->jnt_qposadr[mujoco_joint_id];
      joint_state.mj_vel_adr = mj_model_->jnt_dofadr[mujoco_joint_id];
      joint_state.has_mujoco_joint_state = true;

      auto urdf_joint = urdf_model.getJoint(joint_state.name);
      if (urdf_joint)
      {
        get_joint_limits(urdf_joint, joint_state.joint_limits);
      }
    }

    if (mujoco_actuator_id != -1)
    {
      joint_state.is_mujoco_actuator = true;
      joint_state.mj_actuator_id = mujoco_actuator_id;
      joint_state.mj_actuator_trn_type = mj_model_->actuator_trntype[mujoco_actuator_id];
      joint_state.mj_actuator_trn_id = mj_model_->actuator_trnid[2 * mujoco_actuator_id];

      if (!joint_state.has_mujoco_joint_state)
      {
        joint_state.position = 0.0;
        joint_state.velocity = 0.0;
        joint_state.effort = 0.0;
      }
    }

    joint_states_.at(joint_index) = joint_state;
    JointState &last_joint_state = joint_states_.at(joint_index);

    if (joint.parameters.find("mimic") != joint.parameters.end())
    {
      const auto mimicked_joint = joint.parameters.at("mimic");
      const auto mimicked_joint_it = std::find_if(
        hardware_info.joints.begin(), hardware_info.joints.end(),
        [&mimicked_joint](const hardware_interface::ComponentInfo &info)
        { return info.name == mimicked_joint; });

      if (mimicked_joint_it == hardware_info.joints.end())
      {
        throw std::runtime_error(std::string("Mimicked joint '") + mimicked_joint + "' not found");
      }

      last_joint_state.is_mimic = true;
      last_joint_state.mimicked_joint_index =
        std::distance(hardware_info.joints.begin(), mimicked_joint_it);

      auto param_it = joint.parameters.find("multiplier");
      if (param_it != joint.parameters.end())
      {
        last_joint_state.mimic_multiplier = std::stod(joint.parameters.at("multiplier"));
      }
      else
      {
        last_joint_state.mimic_multiplier = 1.0;
      }
    }

    auto get_initial_value = [](const hardware_interface::InterfaceInfo &interface_info)
    {
      if (!interface_info.initial_value.empty())
      {
        return std::stod(interface_info.initial_value);
      }
      return 0.0;
    };

    for (const auto &state_if : joint.state_interfaces)
    {
      if (state_if.name == hardware_interface::HW_IF_POSITION)
      {
        state_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_POSITION, &last_joint_state.position);
        last_joint_state.position = get_initial_value(state_if);
      }
      else if (state_if.name == hardware_interface::HW_IF_VELOCITY)
      {
        state_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_VELOCITY, &last_joint_state.velocity);
        last_joint_state.velocity = get_initial_value(state_if);
      }
      else if (state_if.name == hardware_interface::HW_IF_EFFORT)
      {
        state_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_EFFORT, &last_joint_state.effort);
        last_joint_state.effort = get_initial_value(state_if);
      }
    }

    auto get_min_value = [](const hardware_interface::InterfaceInfo &interface_info)
    {
      if (!interface_info.min.empty())
      {
        return std::stod(interface_info.min);
      }
      return -1 * std::numeric_limits<double>::max();
    };

    auto get_max_value = [](const hardware_interface::InterfaceInfo &interface_info)
    {
      if (!interface_info.max.empty())
      {
        return std::stod(interface_info.max);
      }
      return std::numeric_limits<double>::max();
    };

    for (const auto &command_if : joint.command_interfaces)
    {
      if (command_if.name.find(hardware_interface::HW_IF_POSITION) != std::string::npos)
      {
        command_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_POSITION, &last_joint_state.position_command);
        last_joint_state.is_position_control_enabled = true;
        last_joint_state.position_command = last_joint_state.position;
        last_joint_state.min_position_command = get_min_value(command_if);
        last_joint_state.max_position_command = get_max_value(command_if);
      }
      else if (command_if.name.find(hardware_interface::HW_IF_VELOCITY) != std::string::npos)
      {
        command_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_VELOCITY, &last_joint_state.velocity_command);
        last_joint_state.is_velocity_control_enabled = true;
        last_joint_state.velocity_command = last_joint_state.velocity;
        last_joint_state.min_velocity_command = get_min_value(command_if);
        last_joint_state.max_velocity_command = get_max_value(command_if);
      }
      else if (command_if.name == hardware_interface::HW_IF_EFFORT)
      {
        command_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_EFFORT, &last_joint_state.effort_command);
        last_joint_state.is_effort_control_enabled = true;
        last_joint_state.effort_command = last_joint_state.effort;
        last_joint_state.min_effort_command = get_min_value(command_if);
        last_joint_state.max_effort_command = get_max_value(command_if);
      }

      if (command_if.name.find("_pid") != std::string::npos)
      {
        last_joint_state.is_pid_enabled = true;
      }
    }

    if (last_joint_state.is_pid_enabled)
    {
      last_joint_state.position_pid = get_pid_gains(joint, hardware_interface::HW_IF_POSITION);
      last_joint_state.velocity_pid = get_pid_gains(joint, hardware_interface::HW_IF_VELOCITY);
    }
  }
}

//#################################################################################################
// Added support for "URDF joint state export for tendon/actuator-controlled MuJoCo models".
//
// Purpose:
//   In this project, some RH8D commands are applied to MuJoCo actuators/tendons
//   instead of directly commanding every physical finger joint.
//
//   Example:
//     - Controller command: pos_index_tendon
//     - But MuJoCo actually moves:
//         Index_Proximal--palmL:1
//         Index_Middle--Index_Proximal
//         Index_Distal--Index_Middle
//
//   Therefore, even if the controller command is sent to a tendon actuator,
//   ROS 2 still needs the real URDF joint states for joint_state_broadcaster,
//   robot_state_publisher, and RViz visualization.
//
// What this function does:
//   1. Loops through all URDF joints.
//   2. Ignores fixed joints because they do not move.
//   3. Finds the matching MuJoCo joint with the same name.
//   4. Stores the MuJoCo qpos/qvel addresses for that joint.
//   5. Creates ROS 2 control state interfaces:
//        joint_name/position
//        joint_name/velocity
//        joint_name/effort
//
// Important:
//   This function only registers the state interfaces.
//   The live values are updated later in read(), where MuJoCo qpos/qvel/qfrc
//   are copied into these stored URDF joint state variables.
//#################################################################################################

void MujocoSystem::register_urdf_joint_states(const urdf::Model &urdf_model)
{
  size_t count = 0;
  for (const auto &joint_pair : urdf_model.joints_)
  {
    const auto &urdf_joint = joint_pair.second;
    if (!urdf_joint) {
      continue;
    }
    if (urdf_joint->type == urdf::Joint::FIXED) {
      continue;
    }
    const auto already_registered = std::find_if(
      joint_states_.begin(), joint_states_.end(),
      [&urdf_joint](const JointState &state) { return state.name == urdf_joint->name; });
    if (already_registered != joint_states_.end()) {
      continue;
    }
    count++;
  }

  urdf_joint_states_.clear();
  urdf_joint_states_.resize(count);

  size_t idx = 0;
  for (const auto &joint_pair : urdf_model.joints_)
  {
    const auto &urdf_joint = joint_pair.second;
    if (!urdf_joint) {
      continue;
    }
    if (urdf_joint->type == urdf::Joint::FIXED) {
      continue;
    }
    const auto already_registered = std::find_if(
      joint_states_.begin(), joint_states_.end(),
      [&urdf_joint](const JointState &state) { return state.name == urdf_joint->name; });
    if (already_registered != joint_states_.end()) {
      continue;
    }

    int mujoco_joint_id = mj_name2id(mj_model_, mjOBJ_JOINT, urdf_joint->name.c_str());
    if (mujoco_joint_id == -1)
    {
      RCLCPP_WARN_STREAM(
        logger_,
        "URDF joint not found in mujoco model for state export: " << urdf_joint->name);
      continue;
    }

    UrdfJointState state;
    state.name = urdf_joint->name;
    state.mj_joint_id = mujoco_joint_id;
    state.mj_pos_adr = mj_model_->jnt_qposadr[mujoco_joint_id];
    state.mj_vel_adr = mj_model_->jnt_dofadr[mujoco_joint_id];

    urdf_joint_states_.at(idx) = state;
    UrdfJointState &last_state = urdf_joint_states_.at(idx);

    state_interfaces_.emplace_back(
      last_state.name, hardware_interface::HW_IF_POSITION, &last_state.position);
    state_interfaces_.emplace_back(
      last_state.name, hardware_interface::HW_IF_VELOCITY, &last_state.velocity);
    state_interfaces_.emplace_back(
      last_state.name, hardware_interface::HW_IF_EFFORT, &last_state.effort);

    idx++;
  }
}
//########## Added support for "URDF joint state export for tendon-driven MuJoCo models" during this project ##########


void MujocoSystem::register_sensors(
  const urdf::Model & /* urdf_model */, const hardware_interface::HardwareInfo &hardware_info)
{
  // Assuming force/torque sensor end with "_fts" in the name,
  // and IMU sensor end with "_imu" in the name
  for (size_t sensor_index = 0; sensor_index < hardware_info.sensors.size(); sensor_index++)
  {
    auto sensor = hardware_info.sensors.at(sensor_index);
    std::string sensor_name = sensor.name;
    sensor_name = sensor_name.substr(0, sensor_name.rfind('_'));

    if (sensor.name.find("_fts") != std::string::npos)
    {
      FTSensorData sensor_data;
      sensor_data.name = sensor_name;
      sensor_data.force.name = sensor_name + "_force";
      sensor_data.torque.name = sensor_name + "_torque";

      int force_sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.force.name.c_str());
      int torque_sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.torque.name.c_str());

      if (force_sensor_id == -1 || torque_sensor_id == -1)
      {
        RCLCPP_ERROR_STREAM(
          logger_,
          "Failed to find force/torque sensor in mujoco model, sensor name: " << sensor.name);
        continue;
      }

      sensor_data.force.mj_sensor_index = mj_model_->sensor_adr[force_sensor_id];
      sensor_data.torque.mj_sensor_index = mj_model_->sensor_adr[torque_sensor_id];

      ft_sensor_data_.push_back(sensor_data);
      auto &last_sensor_data = ft_sensor_data_.back();

      for (const auto &state_if : sensor.state_interfaces)
      {
        if (state_if.name == "force.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.force.data.x());
        }
        else if (state_if.name == "force.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.force.data.y());
        }
        else if (state_if.name == "force.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.force.data.z());
        }
        else if (state_if.name == "torque.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.torque.data.x());
        }
        else if (state_if.name == "torque.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.torque.data.y());
        }
        else if (state_if.name == "torque.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.torque.data.z());
        }
      }
    }
    else if (sensor.name.find("_imu") != std::string::npos)
    {
      IMUSensorData sensor_data;
      sensor_data.name = sensor_name;
      sensor_data.orientation.name = sensor_name + "_quat";
      sensor_data.angular_velocity.name = sensor_name + "_gyro";
      sensor_data.linear_acceleration.name = sensor_name + "_accel";

      int quat_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.orientation.name.c_str());
      int gyro_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.angular_velocity.name.c_str());
      int accel_id =
        mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.linear_acceleration.name.c_str());

      if (quat_id == -1 || gyro_id == -1 || accel_id == -1)
      {
        RCLCPP_ERROR_STREAM(
          logger_, "Failed to find IMU sensor in mujoco model, sensor name: " << sensor.name);
        continue;
      }

      sensor_data.orientation.mj_sensor_index = mj_model_->sensor_adr[quat_id];
      sensor_data.angular_velocity.mj_sensor_index = mj_model_->sensor_adr[gyro_id];
      sensor_data.linear_acceleration.mj_sensor_index = mj_model_->sensor_adr[accel_id];

      imu_sensor_data_.push_back(sensor_data);
      auto &last_sensor_data = imu_sensor_data_.back();

      for (const auto &state_if : sensor.state_interfaces)
      {
        if (state_if.name == "orientation.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.orientation.data.x());
        }
        else if (state_if.name == "orientation.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.orientation.data.y());
        }
        else if (state_if.name == "orientation.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.orientation.data.z());
        }
        else if (state_if.name == "orientation.w")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.orientation.data.w());
        }
        else if (state_if.name == "angular_velocity.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.angular_velocity.data.x());
        }
        else if (state_if.name == "angular_velocity.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.angular_velocity.data.y());
        }
        else if (state_if.name == "angular_velocity.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.angular_velocity.data.z());
        }
        else if (state_if.name == "linear_acceleration.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.linear_acceleration.data.x());
        }
        else if (state_if.name == "linear_acceleration.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.linear_acceleration.data.y());
        }
        else if (state_if.name == "linear_acceleration.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.linear_acceleration.data.z());
        }
      }
    }

    //#################################################################################################
    // Added support for "RH8D fingertip force, tendon length, tendon force, and palm range sensors".
    //
    // Purpose:
    //   The RH8D MuJoCo model contains project-specific sensors that are not handled
    //   by the original generic IMU / force-torque sensor registration logic.
    //
    // Added sensor categories:
    //
    //   1. Fingertip force sensors:
    //        thumb_tip_force
    //        index_tip_force
    //        middle_tip_force
    //        ring_tip_force
    //        small_tip_force
    //
    //      These are 3D vector sensors from MuJoCo force sensors.
    //      They are exposed to ROS 2 control as:
    //        sensor_name/x
    //        sensor_name/y
    //        sensor_name/z
    //
    //   2. Palm range sensor:
    //        palm_range
    //
    //      This is a scalar rangefinder value used to detect whether an object is
    //      close to the palm before closing the hand.
    //      It is exposed as:
    //        palm_range/value
    //
    //   3. Tendon length sensors:
    //        len_index
    //        len_middle
    //        len_ring
    //        len_small
    //        len_thumb
    //
    //      These are scalar tendon position/length sensors.
    //      They are exposed as:
    //        len_*/value
    //
    //   4. Tendon actuator force sensors:
    //        frc_index
    //        frc_middle
    //        frc_ring
    //        frc_thumb
    //
    //      These are scalar actuator force sensors for the tendon actuators.
    //      They are exposed as:
    //        frc_*/value
    //
    // What this block does:
    //   1. Detects RH8D-specific sensors by name.
    //   2. Finds the matching sensor inside the MuJoCo model.
    //   3. Stores the starting index of that sensor inside mj_data_->sensordata[].
    //   4. Creates ROS 2 control state interfaces for the sensor values.
    //
    // Important:
    //   This block only registers the sensor interfaces.
    //   The live sensor values are updated later in read(), where values are copied
    //   from mj_data_->sensordata[] into Vec3SensorData or ScalarSensorData.
    //#################################################################################################
    else if (sensor.name.find("_tip_force") != std::string::npos)
    {
      int sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor.name.c_str());
      if (sensor_id == -1)
      {
        RCLCPP_ERROR_STREAM(logger_, "Failed to find vec3 sensor in mujoco model: " << sensor.name);
        continue;
      }

      Vec3SensorData sensor_data;
      sensor_data.name = sensor.name;
      sensor_data.mj_sensor_index = mj_model_->sensor_adr[sensor_id];

      vec3_sensor_data_.push_back(sensor_data);
      auto &last_sensor_data = vec3_sensor_data_.back();

      for (const auto &state_if : sensor.state_interfaces)
      {
        if (state_if.name == "x")
        {
          state_interfaces_.emplace_back(sensor.name, state_if.name, &last_sensor_data.data.x());
        }
        else if (state_if.name == "y")
        {
          state_interfaces_.emplace_back(sensor.name, state_if.name, &last_sensor_data.data.y());
        }
        else if (state_if.name == "z")
        {
          state_interfaces_.emplace_back(sensor.name, state_if.name, &last_sensor_data.data.z());
        }
      }

      continue;
    }

    else if (
      sensor.name == "palm_range" ||
      sensor.name.rfind("len_", 0) == 0 ||
      sensor.name.rfind("frc_", 0) == 0)
    {
      int sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor.name.c_str());
      if (sensor_id == -1)
      {
        RCLCPP_ERROR_STREAM(logger_, "Failed to find scalar sensor in mujoco model: " << sensor.name);
        continue;
      }

      ScalarSensorData sensor_data;
      sensor_data.name = sensor.name;
      sensor_data.mj_sensor_index = mj_model_->sensor_adr[sensor_id];

      scalar_sensor_data_.push_back(sensor_data);
      auto &last_sensor_data = scalar_sensor_data_.back();

      for (const auto &state_if : sensor.state_interfaces)
      {
        if (state_if.name == "value")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.value);
        }
      }

      continue;
    }
    //########## Added support for "Force Tip sensor and Palm Distance sensor" during this project ##########


  }
}

void MujocoSystem::set_initial_pose()
{
  for (auto &joint_state : joint_states_)
  {
    if (joint_state.is_mujoco_actuator)
    {
      mj_data_->ctrl[joint_state.mj_actuator_id] = joint_state.position;
      if (joint_state.has_mujoco_joint_state)
      {
        mj_data_->qpos[joint_state.mj_pos_adr] = joint_state.position;
      }
    }
    else
    {
      mj_data_->qpos[joint_state.mj_pos_adr] = joint_state.position;
    }
  }
}

void MujocoSystem::get_joint_limits(
  urdf::JointConstSharedPtr urdf_joint, joint_limits::JointLimits &joint_limits)
{
  if (urdf_joint && urdf_joint->limits)
  {
    joint_limits.min_position = urdf_joint->limits->lower;
    joint_limits.max_position = urdf_joint->limits->upper;
    joint_limits.max_velocity = urdf_joint->limits->velocity;
    joint_limits.max_effort = urdf_joint->limits->effort;
  }
}

control_toolbox::Pid MujocoSystem::get_pid_gains(
  const hardware_interface::ComponentInfo &joint_info, std::string command_interface)
{
  double kp, ki, kd, i_max, i_min;
  std::string key;

  key = command_interface + std::string(PARAM_KP);
  if (joint_info.parameters.find(key) != joint_info.parameters.end())
  {
    kp = std::stod(joint_info.parameters.at(key));
  }
  else
  {
    kp = 0.0;
  }

  key = command_interface + std::string(PARAM_KI);
  if (joint_info.parameters.find(key) != joint_info.parameters.end())
  {
    ki = std::stod(joint_info.parameters.at(key));
  }
  else
  {
    ki = 0.0;
  }

  key = command_interface + std::string(PARAM_KD);
  if (joint_info.parameters.find(key) != joint_info.parameters.end())
  {
    kd = std::stod(joint_info.parameters.at(key));
  }
  else
  {
    kd = 0.0;
  }

  bool enable_anti_windup = false;

  key = command_interface + std::string(PARAM_I_MAX);
  if (joint_info.parameters.find(key) != joint_info.parameters.end())
  {
    i_max = std::stod(joint_info.parameters.at(key));
    enable_anti_windup = true;
  }
  else
  {
    i_max = std::numeric_limits<double>::max();
  }

  key = command_interface + std::string(PARAM_I_MIN);
  if (joint_info.parameters.find(key) != joint_info.parameters.end())
  {
    i_min = std::stod(joint_info.parameters.at(key));
    enable_anti_windup = true;
  }
  else
  {
    i_min = std::numeric_limits<double>::lowest();
  }

  return control_toolbox::Pid(kp, ki, kd, i_max, i_min, enable_anti_windup);
}
}  // namespace mujoco_ros2_control

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  mujoco_ros2_control::MujocoSystem, mujoco_ros2_control::MujocoSystemInterface)
