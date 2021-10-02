import math
import os
from gym.spaces import Box
from roman import Tool, JointSpeeds
from mmky.env import RomanEnv
from mmky.tasks.pour.poursim import PourSim
from mmky.tasks.pour.pourreal import PourReal
import yaml

GRASP_HEIGHT = 0.04

class PourEnv(RomanEnv):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
            config = yaml.safe_load(f)
        super().__init__(PourSim, PourReal, config)
        self.action_space = Box(low=-1, high=1, shape=(3,))
        self.workspace = config.get("workspace", [math.pi - 0.5, math.pi + 0.5, 0.25, 0.45])
        self.__joints = self.robot.joint_positions

    def reset(self):
        source_cup_pos, target_cup_pos = list(self.generate_random_xy(*self.workspace) + [0.025] for i in range(2))
        self.scene.reset(source_cup_pos, target_cup_pos)
        (arm_state, had_state) = self.robot.read()
        start = arm_state.tool_pose()
        start[:2] = self.generate_random_xy(*self.workspace)
        self.robot.move(start, max_speed=3, max_acc=1)
        self.robot.open()
        self.robot.pinch(128)
        self.robot.active_force_limit = (None, None)
        self.__z = self.robot.tool_pose[Tool.Z]
        self.__joints = self.robot.joint_positions
        return self._observe()

    def _act(self, action):
        if action[2] == -1:
            self.__place()
        elif action[2] == 1:
            self.__pick()
        else:
            self.__move(*action[:2])

    def __move(self, dx, dy, droll):
        pose = self.robot.tool_pose
        jc = self.robot.joint_positions
        pose = Tool.from_xyzrpy(pose.to_xyzrpy() + [0.04 * dx, 0.04 * dy, 0, 0, 0, 0])
        pose[Tool.Z] = self.__z
        self.robot.move(pose, max_acc=0.0, timeout=0)
        # after the move call, the target_joint_positions is the IK solution
        jt = self.robot.arm.state.target_joint_positions().clone()
        js = self.__joints
        dj = jt - jc
        dsj = js - jc # keep the last joints in the start configuration
        speeds = JointSpeeds(dj[0], dj[1], dj[2], dsj[3], dsj[4], dj[5] + 0.01 * droll)
        self.robot.move(speeds, max_acc=5, timeout=0)

    def __pick(self):
        back = self.robot.tool_pose
        pick_pose = back.clone()
        pick_pose[Tool.Z] = GRASP_HEIGHT
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
