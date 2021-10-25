import os
from gym.spaces import Box
from mmky.env import RomanEnv
from mmky.tasks.stack.stacksim import StackSim
from mmky.tasks.stack.stackreal import StackReal
from mmky import primitives

GRASP_HEIGHT = 0.04
CUBE_COUNT = 2

class StackEnv(RomanEnv):
    def __init__(self, use_2d_action_space=True):
        super().__init__(StackSim, StackReal, os.path.join(os.path.dirname(__file__), 'xyconfig.yaml'))
        self.action_space = Box(low=-1, high=1, shape=(3,))

    def _act(self, action):
        if self.use_2d_action_space:
            force_full_obs = False
            if action[2] == -1:
                if not primitives.place(self.robot, self.working_height + GRASP_HEIGHT):
                    self.end_episode(False)
                force_full_obs = True
            elif action[2] == 1:
                if not primitives.pick(self.robot, self.working_height + GRASP_HEIGHT):
                    self.end_episode(False)
            else:
                primitives.move_dxdy(self.robot, self.working_height, action[0], action[1], 0)
            return force_full_obs
