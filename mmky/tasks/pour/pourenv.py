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
        with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
            config = yaml.safe_load(f)
        super().__init__(PourSim, PourReal, config)
        self.action_space = Box(low=-1, high=1, shape=(3,))

    def reset(self):
        self.last_reward = 0
        while True:
            min_angle_in_rad, max_angle_in_rad = self.workspace_span
            min_dist, max_dist = self.workspace_radius

            sx, sy = self.generate_random_xy(min_angle_in_rad,
                                            (max_angle_in_rad + min_angle_in_rad) / 2 - 0.1,
                                            min_dist,
                                            max_dist)

            tx, ty = self.generate_random_xy((max_angle_in_rad + min_angle_in_rad) / 2 + 0.1,
                                            max_angle_in_rad,
                                            min_dist,
                                            max_dist)

            obs = super().reset(source_cup_pos=[sx, sy, self.workspace_height], target_cup_pos=[tx, ty, self.workspace_height])
            self.__xyzrpy = self.robot.tool_pose.to_xyzrpy()
            objects = obs["world"]
            if not self.robot.open(timeout=2):
                continue
            self.robot.set_hand_mode(GraspMode.NARROW)
            sx, sy, _ = objects["source"]["position"]
            #sx, sy = self.shift(sx, sy, 0.01, 0)
            pick_pose = self._tool_pose_from_xy(sx, sy)
            if not self.robot.move(pick_pose, max_speed=0.5, max_acc=0.5, timeout=10):
                continue
            if not self.__pick():
                continue

            x, y = self.generate_random_xy(*self.workspace_span, *self.workspace_radius)
            start = self._tool_pose_from_xy(x, y)
            if not self.robot.move(start, max_speed=0.5, max_acc=0.5):
                continue

            #self.robot.active_force_limit = (None, None)
            return self._observe()

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



