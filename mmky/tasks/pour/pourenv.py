import math
import os
from gym.spaces import Box
import numpy as np
import yaml

from roman import Tool, Joints, JointSpeeds
from mmky.env import RomanEnv
from mmky.tasks.pour.poursim import PourSim
from mmky.tasks.pour.pourreal import PourReal

GRASP_OFFSET = 0.04

class PourEnv(RomanEnv):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
            config = yaml.safe_load(f)
        super().__init__(PourSim, PourReal, config)
        self.action_space = Box(low=-1, high=1, shape=(3,))
        self.workspace = config.get("workspace", [math.pi - 0.5, math.pi + 0.5, 0.25, 0.45])

    def reset(self):
        source_cup_pos, target_cup_pos = list(self.generate_random_xy(*self.workspace) + [0.025] for i in range(2))
        self.robot.open()
        self.robot.pinch(128)
        self.scene.reset(source_cup_pos, target_cup_pos)
        self.__xyzrpy = self.robot.tool_pose.to_xyzrpy()

        sx, sy, sz = self.scene.get_world_state()[0][:3]
        pick_pose = self._tool_pose_from_xy(sx, sy)
        self.robot.move(pick_pose)
        self.__pick()
        
        x, y = self.generate_random_xy(*self.workspace)
        start = self._tool_pose_from_xy(x, y)
        self.robot.move(start)
        
        self.robot.active_force_limit = (None, None)
        return self._observe()

    def _act(self, action):
        if action[2] == -1:
            self.__place()
        elif action[2] == 1:
            self.__pick()
        else:
            self.__move(*action[:2])

    def _tool_pose_from_xy(self, x, y):
        target = np.array(self.__xyzrpy)
        target[:2] = x, y
        target[5] = math.atan2(y, x) + math.pi/2 #yaw, compensating for robot config offset (base offset is pi, wrist offset from base is -pi/2)
        return Tool.from_xyzrpy(target)

    def __move(self, dx, dy, dr):
        pose = self.robot.tool_pose
        joints = self.robot.joint_positions
        target = self._tool_pose_from_xy(pose[Tool.X] + 0.04 * dx, pose[Tool.Y] + 0.04 * dy)
        self.robot.move(target, max_acc=0, timeout=0) # get_IK
        jtarget = self.robot.arm.state.target_joint_positions().clone() # IK solution
        jtarget[5] = joints[5] + 0.04 * dr
        self.robot.move(jtarget, timeout=0)

    def __pick(self):
        back = self.robot.tool_pose
        pick_pose = back.clone()
        pick_pose[Tool.Z] = -0.06
        self.robot.open()
        self.robot.move(pick_pose)
        self.robot.pinch()
        self.robot.move(back)
        self.__has_object = self.robot.has_object

    def __place(self):
        back = self.robot.tool_pose
        pour_pose = back.clone()
        pour_pose[Tool.Z] = GRASP_HEIGHT
        self.robot.touch(pour_pose)
        self.robot.release(128)
        self.robot.move(back, max_speed=2, max_acc=1)
