import math
import os
from gym.spaces import Box, Dict
import numpy as np
import yaml
from roman import JointSpeeds
from mmky.env import RomanEnv

class SimpleEnv(RomanEnv):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
            config = yaml.safe_load(f)
        super().__init__(config=config)
        self.action_space = Box(low=-1, high=1, shape=(7,))
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(self.obs_res[0], self.obs_res[1], 3), dtype=np.uint8), 
            "proprio": Box(low=-np.inf, high=np.inf, shape=(self.observation_space["arm_state"].shape[0] + self.observation_space["hand_state"].shape[0],))})
        self.robot.pinch()
        self.__last_hand_target = 1

    def _act(self, action):
        self.robot.move(JointSpeeds(*action[:6]), max_speed=3, max_acc=1, timeout=0)
        if action[6] != self.__last_hand_target:
            self.__last_hand_target = action[6]
            pos = min(255, max(action[6] * 255, 0))
            self.robot.pinch(position=pos, timeout=0)

    def _observe(self):
        obs = super()._observe()
        return {"image": obs["cameras"][0],
                "proprio": np.concatenate((obs["arm_state"], obs["hand_state"]))}

        

