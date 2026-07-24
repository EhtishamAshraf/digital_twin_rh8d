import os
import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit

from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ur5e_moveit_config_pkg_name = "ur5e_moveit_config"
    ur5e_mujoco_description_pkg_name = "ur5e_mujoco_description"

    ur5e_mujoco_description_pkg = get_package_share_directory(ur5e_mujoco_description_pkg_name)
    ur5e_moveit_config_pkg = get_package_share_directory(ur5e_moveit_config_pkg_name)

    # Build MoveIt configuration for the UR5e
    moveit_config = (
        MoveItConfigsBuilder("ur5e", package_name=ur5e_moveit_config_pkg_name)
        .robot_description_semantic(file_path="config/ur5e.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Process the xacro file to get the robot description
    xacro_file = os.path.join(
        ur5e_moveit_config_pkg,
        'config',
        'ur5e.urdf.xacro'
    )
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    moveit_robot_description = {'robot_description': doc.toxml()}

    # Define the robot_state_publisher node to publish the robot's state to TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            moveit_robot_description,
            {"use_sim_time": True},
        ],
    )

    # Define the MoveIt move_group node, which provides the main interface for motion planning and execution
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            moveit_config.trajectory_execution,
            {"use_sim_time": True},
            {"planning_plugin": "ompl_interface/OMPLPlanner"},
            {"default_planning_pipeline": "ompl"},
        ],
    )

    # Load ROS2 controllers configuration for the UR5e and RH8D
    ros2_controllers_file = os.path.join(
        ur5e_moveit_config_pkg,
        'config',
        'ur5e_rh8d_ros2_controllers.yaml'
    )

    # Define the path to the combined MuJoCo scene XML file, which includes both the UR5e robot and the RH8D hand
    mujoco_scene_path = os.path.join(
        ur5e_mujoco_description_pkg,
        "mjcf",
        "ur5e_rh8d_scene.xml",
    )

    # Define the mujoco_ros2_control node, which interfaces with the MuJoCo simulator and ROS2 control framework
    node_mujoco_ros2_control = Node(
        package="mujoco_ros2_control",
        executable="mujoco_ros2_control",
        output="screen",
        parameters=[
            moveit_robot_description,
            ros2_controllers_file,
            {"mujoco_model_path": mujoco_scene_path},
            {"use_sim_time": True},
        ],
    )

    # Define the spawner node for the joint state broadcaster, which will be launched first to ensure that the robot's state is being published before any controllers are activated
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # Define the spawner nodes for the arm and hand trajectory controllers, which will be launched sequentially after the joint state broadcaster is up and running
    arm_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )
    hand_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "hand_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # Define the custom nodes for gestures and pick-and-place operations, which will be launched after the controllers are up and running to ensure that they have access to the robot's state and can send commands to the controllers
    gestures_node = Node(
        package="ur5e_mujoco_control",
        executable="gestures",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True}
        ],
    )
    cartesian_pickNplace_node = Node(
        package="ur5e_mujoco_control",
        executable="cartesian_pickNplace",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True}
        ],
    )
    object_pickNplace_node = Node(
        package="ur5e_mujoco_control",
        executable="object_pickNplace",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True}
        ],
    )
    
    object_pick_node = Node(
        package="ur5e_mujoco_control",
        executable="object_pick",
        output="screen",
        parameters=[
            {"use_sim_time": True}
        ],
    )

    # Define the RViz node to visualize the robot and its state
    rviz_config_file = os.path.join(
        ur5e_moveit_config_pkg,
        "config",
        "moveit.rviz",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
    )


    return LaunchDescription(
        [
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=node_mujoco_ros2_control,
                    on_start=[joint_state_broadcaster_spawner],
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=joint_state_broadcaster_spawner,
                    on_exit=[arm_trajectory_controller_spawner],
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=arm_trajectory_controller_spawner,
                    on_exit=[hand_trajectory_controller_spawner],
                )
            ),
            node_mujoco_ros2_control,
            robot_state_publisher,
            move_group_node,
            rviz_node,
            # gestures_node,
            # cartesian_pickNplace_node,
            # object_pickNplace_node,
            object_pick_node,
        ]
    )


