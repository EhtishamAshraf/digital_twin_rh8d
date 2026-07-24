#!/usr/bin/env python3

import math


def clamp_mapping(value, in_min, in_max, out_min, out_max):
    """
    Same behavior as hardware mapping.h:
    if value < in_min  -> out_min
    if value > in_max  -> out_max
    else linear mapping
    """
    if value < in_min:
        return out_min
    if value > in_max:
        return out_max

    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min # mapping formula


def finger_motor_to_staged_joints(motor_tick):
    """
    Hardware-equivalent staged finger mapping.

    Stage 1:
      0 -> 2000 ticks maps proximal/MCP to 0 -> pi/2

    Stage 2:
      2000 -> 3800 ticks maps middle/PIP to 0 -> pi/2

    Stage 3:
      3800 -> 4095 ticks maps distal/DIP to 0 -> pi/4
    """
    proximal = clamp_mapping(motor_tick, 0.0, 2000.0, 0.0, math.pi / 2.0)
    middle = clamp_mapping(motor_tick, 2000.0, 3800.0, 0.0, math.pi / 2.0)
    distal = clamp_mapping(motor_tick, 3800.0, 4095.0, 0.0, math.pi / 4.0)

    return proximal, middle, distal


def print_test_table():
    test_ticks = [0, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3800, 3900, 4000, 4095]

    print("tick, proximal_rad, middle_rad, distal_rad")
    for tick in test_ticks:
        proximal, middle, distal = finger_motor_to_staged_joints(tick)
        print(f"{tick:4.0f}, {proximal: .4f}, {middle: .4f}, {distal: .4f}")


if __name__ == "__main__":
    print_test_table()
