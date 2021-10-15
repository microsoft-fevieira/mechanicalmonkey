import math
import os
from gym.spaces import Box
import numpy as np
import yaml

from roman import Tool, Joints, JointSpeeds, GraspMode
from mmky.env import RomanEnv
from mmky.tasks.pour.poursim import PourSim
from mmky.tasks.pour.pourreal import PourReal

GRASP_OFFSET = 0.07

class PourEnv(RomanEnv):
    def __init__(self):
        super().__init__(PourSim, PourReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))
        self.action_space = Box(low=-1, high=1, shape=(3,))

    

    def _reward(self, obs):
        rew = obs["world"]["ball_data"]["poured"] - self.last_reward
        self.last_reward = obs["world"]["ball_data"]["poured"]
        return rew

    def _done(self, obs):
        return obs["world"]["ball_data"]["remaining"] == 0 or super()._done(obs)

    def _tool_pose_from_xy(self, x, y):
        target = np.array(self.__xyzrpy)
        target[:2] = x, y
        target[5] = math.atan2(y, x) + math.pi / 2 # yaw, compensating for robot config offset (base offset is pi, wrist offset from base is -pi/2)
        return Tool.from_xyzrpy(target)

    def shift(self, x, y, dr, da):
        r0 = math.sqrt(x * x + y * y)
        a0 = math.atan2(y, x)
        r = r0 + dr
        a = a0 + da
        x1 = r * math.cos(a)
        y1 = r * math.sin(a)
        return x1, y1

    def _act(self, action):
        dx, dy, dr = action
        pose = self.robot.tool_pose
        joints = self.robot.joint_positions
        if dx or dy:
            target = self._tool_pose_from_xy(pose[Tool.X] + 0.01 * dx, pose[Tool.Y] + 0.01 * dy)
            jtarget = self.robot.get_inverse_kinematics(target)
        else:
            jtarget = joints.clone()
        jtarget[Joints.WRIST3] = joints[Joints.WRIST3] + 0.3 * dr
        self.robot.move(jtarget, max_speed=0.3, max_acc=1, timeout=0)
        return False # force_full_obs



