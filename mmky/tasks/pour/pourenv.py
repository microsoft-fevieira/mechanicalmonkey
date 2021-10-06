import math
import os
from posixpath import dirname
from gym.spaces import Box
import numpy as np
import yaml

from roman import Tool, Joints, JointSpeeds
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

        self.robot.open()
        self.robot.pinch(128)
        objects = self.scene.reset([sx, sy, self.workspace_height], [tx, ty, self.workspace_height])
        self.__xyzrpy = self.robot.tool_pose.to_xyzrpy()

        sx, sy, _ = objects["source"]["position"]
        #sx, sy = self.shift(sx, sy, 0.01, 0)
        pick_pose = self._tool_pose_from_xy(sx, sy)
        self.robot.move(pick_pose, max_speed=1, max_acc=1)
        self.__pick()

        x, y = self.generate_random_xy(*self.workspace_span, *self.workspace_radius)
        start = self._tool_pose_from_xy(x, y)
        self.robot.move(start)

        #self.robot.active_force_limit = (None, None)
        return self._observe()

    def _act(self, action):
        if action[2] != 0:
            self.__pour(action[2])
        else:
            self.__move(*action)

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

    def __move(self, dx, dy, dr):
        pose = self.robot.tool_pose
        joints = self.robot.joint_positions
        target = self._tool_pose_from_xy(pose[Tool.X] + 0.04 * dx, pose[Tool.Y] + 0.04 * dy)
        self.robot.move(target, max_acc=0.01, timeout=0) # get_IK
        jtarget = self.robot.arm.state.target_joint_positions().clone() # IK solution
        jtarget[5] = joints[5] + 0.04 * dr
        self.robot.move(jtarget, max_speed=0.1, max_acc=1, timeout=0)

    def __pick(self):
        back = self.robot.tool_pose
        pick_pose = back.clone()
        pick_pose[Tool.Z] = self.workspace_height + GRASP_OFFSET
        self.robot.open()
        self.robot.move(pick_pose)
        self.robot.pinch()
        self.robot.move(back)
        self.__has_object = self.robot.has_object

    def __place(self):
        back = self.robot.tool_pose
        release_pose = back.clone()
        release_pose[Tool.Z] = self.workspace_height + GRASP_OFFSET
        self.robot.touch(release_pose)
        self.robot.release(128)
        self.robot.move(back, max_speed=2, max_acc=1)

    def __pour(self, rot):
        back = self.robot.joint_positions
        pour_pose = back.clone()
        pour_pose[Joints.WRIST3] += rot * 3 * math.pi / 5
        self.robot.move(pour_pose)
        self.robot.move(back)
