import math
import os
from gym.spaces import Box
from roman import Tool
from mmky.env import RomanEnv
from mmky.tasks.pour.poursim import PourSim
from mmky.tasks.pour.pourreal import PourReal
import yaml

GRASP_HEIGHT = 0.04
CUBE_COUNT = 2

class PourEnv(RomanEnv):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
            config = yaml.safe_load(f)
        super().__init__(PourSim, PourReal, config)
        self.action_space = Box(low=-1, high=1, shape=(3,))
        self.workspace = config.get("workspace", [math.pi - 0.5, math.pi + 0.5, 0.25, 0.45])

    def reset(self):
        cube_positions = list(self.generate_random_xy(*self.workspace) + (0.025,) for i in range(CUBE_COUNT))
        self.scene.reset(cube_positions)
        (arm_state, had_state) = self.robot.read()
        start = arm_state.tool_pose()
        start[:2] = self.generate_random_xy(*self.workspace)
        self.robot.move(start, max_speed=3, max_acc=1)
        self.robot.open()
        self.robot.pinch(128)
        self.robot.active_force_limit = (None, None)
        self.__z = self.robot.tool_pose[Tool.Z]
        return self._observe()

    def _act(self, action):
        if action[2] == -1:
            self.__place()
        elif action[2] == 1:
            self.__pick()
        else:
            self.__move(*action[:2])

    def __move(self, dx, dy):
        pose = self.robot.tool_pose
        pose = Tool.from_xyzrpy(pose.to_xyzrpy() + [0.01 * dx, 0.01 * dy, 0, 0, 0, 0])
        pose[Tool.Z] = self.__z
        self.robot.move(pose, max_speed=0.5, max_acc=2, timeout=0)

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
