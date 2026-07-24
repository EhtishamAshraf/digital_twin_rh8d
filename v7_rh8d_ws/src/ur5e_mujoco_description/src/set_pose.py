#!/usr/bin/env python3

"""
The purpose of this code is to open the simulation with a specific pose to make use of the MuJoCo GUI to monitor the "Joint angles for the RH8D Hand".

Run the code:
    python3 set_pose.py --scene /home/ehtisham/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/ros2_control_moveit/v6_rh8d_ws/src/ur5e_mujoco_description/mjcf/ur5e_rh8d_scene.xml

"""

import argparse
import sys
import mujoco
import mujoco.viewer

# Function for getting the index of a named joint in the MuJoCo model:
def joint_id(m, name):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint '{name}' not found")
    return jid

# Function for getting the index of a named actuator in the MuJoCo model:
def actuator_id(m, name):
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator '{name}' not found")
    return aid

# Function for setting the position of a specific joint:
def set_joint_qpos(m, d, joint_name, value):
    jid = joint_id(m, joint_name)
    qadr = m.jnt_qposadr[jid]
    d.qpos[qadr] = float(value)

# Function for setting the target of a specific actuator:
def set_actuator_ctrl(m, d, actuator_name, value):
    aid = actuator_id(m, actuator_name)
    d.ctrl[aid] = float(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()

    # Loading MuJoCo model and data
    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)

    # Reset simulation state to default before manually assigning pose
    mujoco.mj_resetData(m, d)

    # UR5e and RH8D startup pose for cylinder
    arm_pose = {
        "shoulder_pan_joint":  1.4927,
        "shoulder_lift_joint": -0.8678,
        "elbow_joint":         -2.43,
        "wrist_1_joint":       -0.6596,
        "wrist_2_joint":       -0.0347,
        "wrist_3_joint":       0.126,
    }
    hand_pose = {
        "forearm:1--base:1": 0.0,
        "palm_axis:1--forearm:1": 0.0,
        "palm_assembly:1--palm_axis:1": 0.0,
        "Thumb_axis--palm_assembly:1": 1.57,
        "Thumb_Methacarpal--Thumb_axis": 0.0,
        "Index_Proximal--palm_assembly:1": 0.0,
        "Middle_Proximal--palm_assembly:1": 0.0,
        "Ring_Proximal--palm_assembly:1": 0.0,
    }

    # Set joints for UR5e and RH8D
    for jname, q in arm_pose.items():
        set_joint_qpos(m, d, jname, q)
    for jname, q in hand_pose.items():
        set_joint_qpos(m, d, jname, q)

    # Set actuator targets to the same pose for UR5e and RH8D
    # This is important because if ctrl targets are not updated, the robot can move back to the previous default actuator target.
    set_actuator_ctrl(m, d, "shoulder_pan", arm_pose["shoulder_pan_joint"])
    set_actuator_ctrl(m, d, "shoulder_lift", arm_pose["shoulder_lift_joint"])
    set_actuator_ctrl(m, d, "elbow", arm_pose["elbow_joint"])
    set_actuator_ctrl(m, d, "wrist_1", arm_pose["wrist_1_joint"])
    set_actuator_ctrl(m, d, "wrist_2", arm_pose["wrist_2_joint"])
    set_actuator_ctrl(m, d, "wrist_3", arm_pose["wrist_3_joint"])
    set_actuator_ctrl(m, d, "pos_forearm", hand_pose["forearm:1--base:1"])
    set_actuator_ctrl(m, d, "pos_palm_axis", hand_pose["palm_axis:1--forearm:1"])
    set_actuator_ctrl(m, d, "pos_palm_assembly", hand_pose["palm_assembly:1--palm_axis:1"])
    set_actuator_ctrl(m, d, "pos_thumb_axis", hand_pose["Thumb_axis--palm_assembly:1"])
    set_actuator_ctrl(m, d, "pos_thumb_tendon", hand_pose["Thumb_Methacarpal--Thumb_axis"])
    set_actuator_ctrl(m, d, "pos_index_tendon", hand_pose["Index_Proximal--palm_assembly:1"])
    set_actuator_ctrl(m, d, "pos_middle_tendon", hand_pose["Middle_Proximal--palm_assembly:1"])
    set_actuator_ctrl(m, d, "pos_ring_small_tendon", hand_pose["Ring_Proximal--palm_assembly:1"])

    # Clearing startup velocities so the model does not start with leftover motion
    d.qvel[:] = 0.0

    # Recompute MuJoCo state after manually changing qpos and ctrl
    mujoco.mj_forward(m, d)

    # Launch viewer and keep simulation running
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running():
                mujoco.mj_step(m, d)
                try:
                    viewer.sync()
                except KeyboardInterrupt:
                    break
    except KeyboardInterrupt:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()