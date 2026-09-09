#!/usr/bin/env python3

from pathlib import Path
import shutil
import re

WS = Path.home() / "Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/digital_twin/v7_rh8d_ws"
ADAPTER = WS / "rh8d_digital_twin_tools/rh8d_sim_hardware_command_adapter.py"

backup = ADAPTER.with_suffix(".py.before_sticky_hold_v5.bak")
shutil.copy2(ADAPTER, backup)

text = ADAPTER.read_text()

replacements = {
    "CONTACT_CLOSE_RATE_TICKS = 180.0": "CONTACT_CLOSE_RATE_TICKS = 120.0",
    "HARD_FORCE_N = 32.0": "HARD_FORCE_N = 45.0",
    "EXTREME_FORCE_N = 55.0": "EXTREME_FORCE_N = 75.0",
    "HOLD_FORCE_MAX_N = 25.0": "HOLD_FORCE_MAX_N = 35.0",
    "HOLD_RELEASE_HIGH_N = 35.0": "HOLD_RELEASE_HIGH_N = 55.0",
    "HOLD_DWELL_SEC = 0.45": "HOLD_DWELL_SEC = 0.80",
    "PUBLISH_DEADBAND_RAD = 0.002": "PUBLISH_DEADBAND_RAD = 0.006",
}

for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)
    else:
        print(f"WARNING: did not find constant: {old}")

new_function = r'''    def update_one_motor_with_hold(self, motor_name, idx, force_name):
        force = self.filtered_force.get(force_name, 0.0)
        target = self.desired_hw[idx]
        current = self.current_hw[idx]

        opening = target < current - 1.0

        if opening:
            self.clear_motor_hold(motor_name)
            self.current_hw[idx] = move_towards(
                current,
                target,
                FREE_CLOSE_RATE_TICKS * DT,
            )
            return

        # Main v5 fix:
        # If hold is already latched, keep it sticky.
        # Do not clear it just because force briefly crosses HARD_FORCE_N.
        if self.hold_latched[motor_name]:
            if force < HOLD_RELEASE_LOW_N:
                # Contact may be temporarily lost. Do not release immediately.
                self.hold_counter[motor_name] += 1

                if self.hold_counter[motor_name] >= int(0.7 * CONTROL_HZ):
                    self.clear_motor_hold(motor_name)
                    self.current_hw[idx] = move_towards(
                        current,
                        target,
                        CONTACT_CLOSE_RATE_TICKS * DT,
                    )
                else:
                    self.current_hw[idx] = self.hold_hw[motor_name]
                return

            if force > HOLD_RELEASE_HIGH_N:
                # Only release/relax for genuinely high force.
                self.clear_motor_hold(motor_name)
                self.current_hw[idx] = max(
                    0.0,
                    current - 0.5 * RELAX_RATE_TICKS * DT,
                )
                return

            # Stable grasp region:
            # Hold the virtual motor target exactly.
            self.hold_counter[motor_name] = 0
            self.current_hw[idx] = self.hold_hw[motor_name]
            return

        # Not currently holding.
        if force >= EXTREME_FORCE_N:
            self.clear_motor_hold(motor_name)
            self.current_hw[idx] = max(
                0.0,
                current - RELAX_RATE_TICKS * DT,
            )
            return

        if force >= HARD_FORCE_N:
            self.clear_motor_hold(motor_name)
            self.current_hw[idx] = max(
                0.0,
                current - 0.5 * RELAX_RATE_TICKS * DT,
            )
            return

        near_target = abs(target - current) <= HOLD_NEAR_TARGET_TICKS
        force_good = HOLD_FORCE_MIN_N <= force <= HOLD_FORCE_MAX_N

        if force_good and near_target:
            self.hold_counter[motor_name] += 1

            if self.hold_counter[motor_name] >= HOLD_DWELL_LOOPS:
                self.hold_latched[motor_name] = True
                self.hold_hw[motor_name] = current
                self.current_hw[idx] = current
                self.get_logger().info(
                    f"HOLD latched for {motor_name}: "
                    f"ticks={current:.0f}, force={force:.2f} N"
                )
                return
        else:
            self.hold_counter[motor_name] = 0

        if force >= CONTACT_FORCE_N:
            self.current_hw[idx] = move_towards(
                current,
                target,
                CONTACT_CLOSE_RATE_TICKS * DT,
            )
        else:
            self.current_hw[idx] = move_towards(
                current,
                target,
                FREE_CLOSE_RATE_TICKS * DT,
            )

'''

pattern = re.compile(
    r'    def update_one_motor_with_hold\(self, motor_name, idx, force_name\):\n.*?\n    def update_virtual_motor_ticks',
    re.DOTALL,
)

text, count = pattern.subn(new_function + "    def update_virtual_motor_ticks", text, count=1)

if count != 1:
    raise RuntimeError("Could not replace update_one_motor_with_hold(). Check adapter file manually.")

ADAPTER.write_text(text)

print(f"Backup saved: {backup}")
print("Applied sticky-hold v5 patch.")
print("")
print("Main changes:")
print("  HOLD is now sticky.")
print("  HOLD no longer releases at 30–40 N.")
print("  Release only happens below low force for ~0.7 s or above 55 N.")
print("  Publish deadband increased to reduce chatter.")
