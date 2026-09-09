#!/usr/bin/env bash
set -e

WS="$HOME/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/digital_twin/v7_rh8d_ws"

cd "$WS"

source /opt/ros/humble/setup.bash
source install/setup.bash

echo "Starting RH8D MuJoCo digital twin..."

ros2 launch ur5e_moveit_config ur5e_rh8d_mujoco_bringup.launch.py &
SIM_PID=$!

cleanup()
{
  echo ""
  echo "Stopping RH8D digital twin..."
  kill $SIM_PID $ADAPTER_PID $SENSOR_PID 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Waiting for MuJoCo/controller manager..."
for i in {1..40}; do
  if ros2 service list | grep -q "/controller_manager/list_controllers"; then
    break
  fi
  sleep 1
done

echo "Waiting for /dynamic_joint_states..."
for i in {1..40}; do
  if ros2 topic list | grep -q "^/dynamic_joint_states$"; then
    break
  fi
  sleep 1
done

echo "Starting hardware command adapter..."
python3 rh8d_digital_twin_tools/rh8d_sim_hardware_command_adapter.py &
ADAPTER_PID=$!

echo "Starting sensor bridge..."
python3 rh8d_digital_twin_tools/rh8d_sim_sensor_bridge.py &
SENSOR_PID=$!

echo ""
echo "RH8D digital twin is running."
echo ""
echo "Hardware-like command topic:"
echo "  /rh8dr/hand_position_controller/commands"
echo ""
echo "Hardware-like sensor topics:"
echo "  /rh8dr/fts_broadcaster/fingertip_force_sensors"
echo "  /rh8dr/tof_broadcaster/distance_measured"
echo ""
echo "Press Ctrl+C to stop everything."

wait
