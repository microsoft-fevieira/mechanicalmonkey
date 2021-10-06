import math
import os
from gym.spaces import Box
from roman import Tool
from mmky.env import RomanEnv
from mmky.tasks.stack.stacksim import StackSim
from mmky.tasks.stack.stackreal import StackReal
import yaml

GRASP_HEIGHT = 0.04
CUBE_COUNT = 2

class StackEnv(RomanEnv):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
            config = yaml.safe_load(f)
        super().__init__(StackSim, StackReal, config)
        self.action_space = Box(low=-1, high=1, shape=(3,))
        
    def reset(self):
        
        cube_positions = list(self.generate_random_xy(*self.workspace_span, *self.workspace_radius) + [self.workspace_height + 0.025] 
                              for i in range(CUBE_COUNT))
        self.scene.reset(cube_positions)
        (arm_state, had_state) = self.robot.read()
        start = arm_state.tool_pose()
        start[:2] = self.generate_random_xy(*self.workspace_span, *self.workspace_radius)
        self.robot.move(start, max_speed=3, max_acc=1)
        self.robot.open()
        self.robot.pinch(128)
        self.robot.active_force_limit = (None, None)
        self.__z = self.robot.tool_pose[Tool.Z]
        obs = self._observe()
        assert(len(obs["world"]) == 2)
        return obs

    def _reward(self, obs):
        return CUBE_COUNT - len(obs["world"]) # TODO: include a stack height check

    def _act(self, action):
        if action[2] == -1:
            return self.__place()
        elif action[2] == 1:
            return self.__pick()
        else:
            return self.__move(*action[:2])

    def __move(self, dx, dy):
        pose = self.robot.tool_pose
        pose = Tool.from_xyzrpy(pose.to_xyzrpy() + [0.01 * dx, 0.01 * dy, 0, 0, 0, 0])
        pose[Tool.Z] = self.__z
        self.robot.move(pose, max_speed=0.1, max_acc=1, timeout=0)
        return False

    def __pick(self):
        back = self.robot.tool_pose
        pick_pose = back.clone()
        pick_pose[Tool.Z] = self.workspace_height + GRASP_HEIGHT
        self.robot.open()
        self.robot.move(pick_pose)
        self.robot.pinch()
        self.robot.move(back)
        self.__has_object = self.robot.has_object
        return False

    def __place(self):
        back = self.robot.tool_pose
        stack_pose = back.clone()
        stack_pose[Tool.Z] = self.workspace_height + GRASP_HEIGHT
        self.robot.touch(stack_pose)
        self.robot.release()
        self.robot.move(back, max_speed=2, max_acc=1)
        return True
