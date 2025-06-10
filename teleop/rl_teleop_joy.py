#!/usr/bin/env python3

import numpy as np
import time
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from rl_policy_inference import RLPolicyInference

from multiprocessing import shared_memory, Array, Lock
import threading

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.robot_arm import G1_29_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from scipy.spatial.transform import Rotation as R

POLICY_PATH = os.path.join(parent_dir, "rl_assets/policy_only_target_wide_range.onnx")
JOINT_YAML_PATH = os.path.join(parent_dir, "rl_assets/joint_order.yml")
SPACES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
]


class JoystickWrapper(Node):
    def __init__(self, joy_topic="/joy"):
        super().__init__("joystick_teleop")

        self.head_rmat = np.eye(3)
        self.left_wrist = np.eye(4)
        self.right_wrist = np.eye(4)

        self.joy_data = None

        self.left_lever_base_pose = np.eye(4)
        self.left_lever_base_pose[0, 3] = 0.2
        self.left_lever_base_pose[1, 3] = 0.2
        self.left_lever_base_pose[2, 3] = 0.0

        left_roll = 20.0 * np.pi / 180.0
        left_pitch = 10.0 * np.pi / 180.0

        left_roll_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(left_roll), -np.sin(left_roll)],
                [0, np.sin(left_roll), np.cos(left_roll)],
            ]
        )
        left_pitch_mat = np.array(
            [
                [np.cos(left_pitch), 0, np.sin(left_pitch)],
                [0, 1, 0],
                [-np.sin(left_pitch), 0, np.cos(left_pitch)],
            ]
        )
        self.left_lever_base_pose[0:3, 0:3] = np.dot(left_roll_mat, left_pitch_mat)

        self.right_lever_base_pose = np.eye(4)
        self.right_lever_base_pose[0, 3] = 0.2
        self.right_lever_base_pose[1, 3] = -0.2
        self.right_lever_base_pose[2, 3] = 0.0

        right_roll = -20.0 * np.pi / 180.0
        right_pitch = 10.0 * np.pi / 180.0

        right_roll_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(right_roll), -np.sin(right_roll)],
                [0, np.sin(right_roll), np.cos(right_roll)],
            ]
        )
        right_pitch_mat = np.array(
            [
                [np.cos(right_pitch), 0, np.sin(right_pitch)],
                [0, 1, 0],
                [-np.sin(right_pitch), 0, np.cos(right_pitch)],
            ]
        )
        self.right_lever_base_pose[0:3, 0:3] = np.dot(right_roll_mat, right_pitch_mat)

        self.lever_length = 0.15

        self.joy_sub = self.create_subscription(
            Joy,
            joy_topic,
            self.joyCb,
            10,
        )

    def joyCb(self, msg):
        self.joy_data = msg
        self.updateWristPoses()

    def updateWristPoses(self):
        if self.joy_data is None:
            return
        print("Updating wrist poses...")

        left_x = self.joy_data.axes[0] if len(self.joy_data.axes) >= 8 else 0.0
        left_y = self.joy_data.axes[1] if len(self.joy_data.axes) >= 8 else 0.0

        right_x = self.joy_data.axes[3] if len(self.joy_data.axes) >= 8 else 0.0
        right_y = self.joy_data.axes[4] if len(self.joy_data.axes) >= 8 else 0.0

        # Mapping joystick values to lever angles
        left_pitch = left_y * 20.0 * np.pi / 180.0
        left_roll = left_x * 20.0 * np.pi / 180.0

        right_pitch = right_y * 20.0 * np.pi / 180.0
        right_roll = right_x * 20.0 * np.pi / 180.0

        left_lever_tip = self.calculateLeverTipPose(
            self.left_lever_base_pose, left_roll, left_pitch, self.lever_length
        )

        right_lever_tip = self.calculateLeverTipPose(
            self.right_lever_base_pose, right_roll, right_pitch, self.lever_length
        )

        self.left_wrist = left_lever_tip
        self.right_wrist = right_lever_tip

    def calculateLeverTipPose(self, base_pose, roll, pitch, length):
        roll_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        pitch_mat = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        rotation = np.dot(roll_mat, pitch_mat)

        base_position = base_pose[0:3, 3].copy()
        base_orientation = base_pose[0:3, 0:3].copy()

        new_orientation = np.dot(rotation, base_orientation)

        lever_tip = np.eye(4)
        lever_tip[0:3, 0:3] = new_orientation

        tip_offset = np.dot(new_orientation, np.array([0, 0, length]))
        lever_tip[0:3, 3] = base_position + tip_offset

        return lever_tip

    def getData(self):
        self.updateWristPoses()

        left_wrist_adjusted = self.left_wrist.copy()
        right_wrist_adjusted = self.right_wrist.copy()

        return left_wrist_adjusted, right_wrist_adjusted


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joy_topic", type=str, default="/joy", help="Joy topic for commands"
    )
    parser.add_argument(
        "--frequency", type=float, default=30.0, help="control loop frequency"
    )
    args = parser.parse_args()
    print(f"args:{args}")

    # Initialize RL policy inference
    rl_inference = RLPolicyInference(POLICY_PATH, 0.5, JOINT_YAML_PATH, SPACES, SPACES)

    # Initialize joystick wrapper
    joystick_wrapper = JoystickWrapper(joy_topic=args.joy_topic)

    arm_ctrl = G1_29_ArmController()
    arm_ik = G1_29_ArmIK()
    arm_ctrl.ctrl_dual_arm_go_home()

    try:
        user_input = input("Please enter the start signal (enter )")
        if user_input.lower() == "r":
            print("Starting teleoperation...")
            arm_ctrl.speed_gradual_max()

            running = True
            timer_period = 1.0 / args.frequency
            while running:
                start_time = time.time()

                rclpy.spin_once(joystick_wrapper, timeout_sec=0.001)

                time_ik_start = time.time()

                left_wrist, right_wrist = joystick_wrapper.getData()

                left_xyz = left_wrist[:3, 3]
                right_xyz = right_wrist[:3, 3]

                left_rot = left_wrist[:3, :3]
                left_quat = R.from_matrix(left_rot).as_quat(scalar_first=True)
                right_rot = right_wrist[:3, :3]
                right_quat = R.from_matrix(right_rot).as_quat(scalar_first=True)

                current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                sol_q = rl_inference.run(
                    q=current_lr_arm_q,
                    dq=current_lr_arm_dq,
                    target=np.concatenate([left_xyz, right_xyz, left_quat, right_quat]),
                )

                sol_tauff = np.zeros_like(sol_q)

                time_ik_end = time.time()
                time_ik = time_ik_end - time_ik_start
                print("Time required for IK: %f", time_ik)
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

                elapsed_time = time.time() - start_time
                if elapsed_time < timer_period:
                    time.sleep(timer_period - elapsed_time)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        print("Finally, exiting program")
        exit(0)


if __name__ == "__main__":
    main()
