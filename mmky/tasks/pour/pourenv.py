import math
import os
from posixpath import dirname
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
        self.robot.open()
        self.robot.set_hand_mode(GraspMode.NARROW)
        self.robot.grasp(128)
        sx, sy, _ = objects["source"]["position"]
        #sx, sy = self.shift(sx, sy, 0.01, 0)
        pick_pose = self._tool_pose_from_xy(sx, sy)
        self.robot.move(pick_pose, max_speed=0.5, max_acc=0.5)
        self.__pick()

        x, y = self.generate_random_xy(*self.workspace_span, *self.workspace_radius)
        start = self._tool_pose_from_xy(x, y)
        self.robot.move(start, max_speed=0.5, max_acc=0.5)

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
            self.robot.move(target, max_speed=0.3, max_acc=1, timeout=0) # get_IK
            jtarget = self.robot.arm.state.target_joint_positions().clone() # IK solution
        else:
            jtarget = joints.clone()
        jtarget[Joints.WRIST3] = joints[Joints.WRIST3] + 0.3 * dr
        self.robot.move(jtarget, max_speed=0.3, max_acc=1, timeout=0)

    def __pick(self):
        back = self.robot.tool_pose
        pick_pose = back.clone()
        pick_pose[Tool.Z] = self.workspace_height + GRASP_OFFSET
        self.robot.open()
        self.robot.move(pick_pose, max_speed=0.5, max_acc=0.5)
        self.robot.grasp()
        self.robot.move(back, max_speed=0.5, max_acc=0.5)
        self.__has_object = self.robot.has_object

    def __place(self):
        back = self.robot.tool_pose
        release_pose = back.clone()
        release_pose[Tool.Z] = self.workspace_height + GRASP_OFFSET
        self.robot.touch(release_pose)
        self.robot.release(128)
        self.robot.move(back, max_speed=0.5, max_acc=0.5)

