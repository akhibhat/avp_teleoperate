import onnxruntime as ort
import yaml
import numpy as np


class RLPolicyInference:
    def __init__(
        self,
        policy_path: str,
        action_scale: float,
        joint_yaml_path: str,
        obs_space: list,
        action_space: list,
    ):
        self.onnx_policy = ort.InferenceSession(policy_path)
        self.obs_space = obs_space
        self.action_space = action_space
        self.action_scale = action_scale
        self.parse_joints_yaml(joint_yaml_path)
        self.previous_actions = np.zeros(len(self.action_space), dtype=np.float32)

    def run(self, q, dq, target_pos):
        q_isaacsim = self.convert_to_isaacsim(q, self.obs_space)
        dq_isaacsim = self.convert_to_isaacsim(dq, self.obs_space)
        model_input = (
            np.concatenate([q_isaacsim, dq_isaacsim, self.previous_actions, target_pos])
            .reshape(1, -1)
            .astype(np.float32)
        )
        action = (
            self.onnx_policy.run(
                [self.onnx_policy.get_outputs()[0].name],
                {self.onnx_policy.get_inputs()[0].name: model_input},
            )[0].squeeze()
            * self.action_scale
        )

        self.previous_actions = action.copy()

        action_unitree = self.convert_to_unitree(action, self.action_space)
        return action_unitree

    def parse_joints_yaml(self, path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        self.isaacsim_joints = data["isaacsim"]
        self.unitree_joints = data["unitree"]

    def convert_to_isaacsim(self, x, trimmed_joints=None):
        if trimmed_joints is not None:
            effective_isaacsim_joints = [
                joint for joint in self.isaacsim_joints if joint in trimmed_joints
            ]
            effective_unitree_joints = [
                joint for joint in self.unitree_joints if joint in trimmed_joints
            ]
        else:
            effective_isaacsim_joints = self.isaacsim_joints
            effective_unitree_joints = self.unitree_joints

        isaacsim_inputs = np.zeros(len(effective_isaacsim_joints))
        for index, joint in enumerate(effective_isaacsim_joints):
            isaacsim_inputs[index] = x[effective_unitree_joints.index(joint)]
        return isaacsim_inputs

    def convert_to_unitree(self, x, trimmed_joints=None):
        if trimmed_joints is not None:
            effective_unitree_joints = [
                joint for joint in self.unitree_joints if joint in trimmed_joints
            ]
            effective_isaacsim_joints = [
                joint for joint in self.isaacsim_joints if joint in trimmed_joints
            ]
        else:
            effective_unitree_joints = self.unitree_joints
            effective_isaacsim_joints = self.isaacsim_joints

        unitree_inputs = np.zeros(len(effective_unitree_joints))
        for index, joint in enumerate(effective_unitree_joints):
            unitree_inputs[index] = x[effective_isaacsim_joints.index(joint)]
        return unitree_inputs


if __name__ == "__main__":
    policy_path = "/home/ubuntu/Prometheus/exported_policies/left_hand_ik.onnx"
    joint_yaml_path = "/home/ubuntu/Prometheus/assets/g1_29dof_lock_waist_with_inspire_hand_DFQ/joint_order.yml"
    spaces = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
    ]
    rl_inference = RLPolicyInference(policy_path, 0.5, joint_yaml_path, spaces, spaces)

    print(
        rl_inference.run(
            q=np.array([-0.2, 0.42, -0.23, 0.87, 0.16, 0.35, -0.16]),
            dq=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            target_pos = np.array([0.2, 0.1, 0.2])
        )
    )
