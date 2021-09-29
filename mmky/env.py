import gym
from gym.spaces import Box, Dict, Tuple
import numpy as np
import math
import random
from roman import Robot, SimScene
from roman.ur import arm
from roman.rq import hand
from mmky.realscene import RealScene
import cv2


MAX_OBJECTS_IN_SCENE = 10

class RomanEnv(gym.Env):
    def __init__(self, simscenefn=SimScene, realscenefn=RealScene, config={}):
        super().__init__()
        self.config = config
        use_sim = config.get("use_sim", True)
        robot_config = config.get("robot_config", {})
        self.robot = Robot(use_sim=use_sim, **robot_config).connect()
        self.obs_res = config.get("obs_res", (84, 84))
        if use_sim:
            sim_config = config.get("sim_scene", {})
            self.scene = simscenefn(robot=self.robot, obs_res=self.obs_res, **sim_config).connect()
        else:
            real_config = config.get("sim_scene", {})
            self.scene = realscenefn(robot=self.robot, obs_res=self.obs_res, **real_config).connect()

        camera_count = self.scene.get_camera_count()
        self.observation_space = Dict({
            "cameras": Tuple(camera_count * [Box(low=0, high=255, shape=(self.obs_res[0], self.obs_res[1], 3), dtype=np.uint8)]),
            "world": Box(low=-2, high=2, shape=(MAX_OBJECTS_IN_SCENE,)),
            "arm_state": Box(low=-np.inf, high=np.inf, shape=(arm.State._BUFFER_SIZE,)),
            "hand_state": Box(low=-np.inf, high=np.inf, shape=(hand.State._BUFFER_SIZE,)),
            "last_arm_cmd": Box(low=-np.inf, high=np.inf, shape=(arm.Command._BUFFER_SIZE,)),
            "last_hand_cmd": Box(low=-np.inf, high=np.inf, shape=(hand.Command._BUFFER_SIZE,))})

        self.action_space = Dict({
            "arm": Box(low=-np.inf, high=np.inf, shape=(arm.Command._BUFFER_SIZE,)),
            "hand": Box(low=-np.inf, high=np.inf, shape=(hand.Command._BUFFER_SIZE,))})

    def seed(seed=None):
        """Sets the seed for this env's random number generator."""
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        self.scene.reset()
        return self._observe()

    def step(self, action):
        self._act(action)
        obs = self._observe()
        rew = self._reward(obs)
        done = self._done(obs)
        return obs, rew, done, {}

    def render(self, mode='human'):
        (img, _) = self.scene.get_camera_images()[0]
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow("camera observation", img)
            cv2.waitKey(1)

    def _observe(self):
        arm_state, hand_state = self.robot.read()
        last_arm_cmd, last_hand_cmd = self.robot.last_command()
        images = self.scene.get_camera_images()
        world = self.scene.get_world_state()
        return {"cameras": images,
                "world": world,
                "arm_state": arm_state,
                "hand_state": hand_state,
                "last_arm_cmd": last_arm_cmd,
                "last_hand_cmd": last_hand_cmd}

    def _act(self, action):
        #self.robot.execute(action[0], action[1])
        pass

    def _reward(self, obs):
        return 0

    def _done(self, obs):
        return False

    @staticmethod
    def generate_random_xy(min_angle_in_rad, max_angle_in_rad, min_dist, max_dist):
        # Sample a random distance from the coordinate origin (i.e., arm base) and a random angle.
        dist = min_dist + random.random() * (max_dist - min_dist)
        angle = min_angle_in_rad + random.random() * (max_angle_in_rad - min_angle_in_rad)
        return dist * math.cos(angle), dist * math.sin(angle)
