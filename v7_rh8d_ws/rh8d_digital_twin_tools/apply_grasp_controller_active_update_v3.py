#!/usr/bin/env python3

from pathlib import Path
import shutil
import re

WS = Path.home() / "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/digital_twin/v7_rh8d_ws"
CTRL = WS / "rh8d_digital_twin_tools/rh8d_sim_grasping_controller.py"

backup = CTRL.with_suffix(".py.before_active_update_v3.bak")
shutil.copy2(CTRL, backup)

text = CTRL.read_text()

new_force_control_step = r'''    def force_control_step(self):
        # v3 fix:
        # If additional fingers touch after FORCE_CONTROL starts,
        # add them to the controlled set. Otherwise those fingers can
        # keep pushing passively with high force and the P controller
        # cannot reduce their force.
        current_contacts = self.contacts()

        for finger in current_contacts:
            if finger not in self.active_fingers:
                self.active_fingers.append(finger)
                self.get_logger().info(
                    f"Added late contact to active force control: {finger}"
                )

        # Keep order stable and only use supported control fingers.
        self.active_fingers = [
            finger for finger in CONTROL_FINGERS
            if finger in self.active_fingers
        ]

        for finger in self.active_fingers:
            measured_force = self.filtered_force[finger]
            desired_force = self.desired_force[finger]

            force_error = desired_force - measured_force

            command = self.kp * force_error
            command = clamp(
                command,
                -self.constant_velocity,
                self.constant_velocity,
            )

            self.apply_integrator_command(finger, command)
            self.publish_integrator_debug(finger, command, force_error)

        self.publish_motor_command()

'''

pattern = re.compile(
    r'    def force_control_step\(self\):\n.*?\n    def control_loop\(self\):',
    re.DOTALL,
)

text, count = pattern.subn(
    new_force_control_step + "    def control_loop(self):",
    text,
    count=1,
)

if count != 1:
    raise RuntimeError("Could not patch force_control_step(). Check the file manually.")

CTRL.write_text(text)

print(f"Backup saved: {backup}")
print("Applied grasp controller active-update v3.")
print("")
print("Main change:")
print("  New contacts during FORCE_CONTROL are added to active_fingers.")
