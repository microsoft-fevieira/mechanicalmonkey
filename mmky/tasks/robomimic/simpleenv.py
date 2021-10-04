import os
from gym import Wrapper
from gym.spaces import Box, Dict

import numpy as np
import yaml
from roman import JointSpeeds, Tool
from mmky.env import RomanEnv

class SimpleEnv(Wrapper):
    def __init__(self, envfn=RomanEnv):
        with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
            config = yaml.safe_load(f)
        env = envfn(config=config)
        super().__init__(env)
        self.action_space = Box(low=-1, high=1, shape=(7,))
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(self.obs_res[0], self.obs_res[1], 3), dtype=np.uint8),
            "proprio": Box(low=-np.inf, high=np.inf, shape=(37,))})
        self.robot.pinch()
        self.__last_hand_target = 1

    def reset(self):
        obs = self.env.reset()
        obs = self.__convert_obs(obs)
        return obs

    def step(self, action):
        self.env.robot.move(JointSpeeds(*action[:6]), max_speed=1, max_acc=0.5, timeout=0)
        if action[6] != self.__last_hand_target:
            self.__last_hand_target = action[6]
            pos = min(255, max(action[6] * 255, 0))
            self.env.robot.pinch(position=pos, timeout=0)
        obs = self.env._observe()
        obs = self.__convert_obs(obs)
        rew = self.env._reward(obs)
        done = self.env._done(obs)
        return obs, rew, done, {}

    def __convert_obs(self, obs):
        joint_pos_cos = np.cos(obs["arm_state"].joint_positions())
        joint_pos_sin = np.sin(obs["arm_state"].joint_positions())
        joint_vel = np.sin(obs["arm_state"].joint_speeds())
        eef = obs["arm_state"].tool_pose()
        eef_pos = eef.position()
        eef_quat = eef.orientation()
        gripper_qpos = [obs["hand_state"].position_A(), 0, obs["hand_state"].position_B(), 0, obs["hand_state"].position_C(), 0]
        gripper_qvel = [0, 0, 0, 0, 0, 0]
        return {"image": obs["cameras"][0],
                "proprio": np.concatenate((joint_pos_cos,
                                          joint_pos_sin,
                                          joint_vel,
                                          eef_pos,
                                          eef_quat,
                                          gripper_qpos,
                                          gripper_qvel))}



