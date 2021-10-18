import os
from cv2 import sepFilter2D
from gym.spaces import Box
from roman import Tool, Joints
from mmky.env import RomanEnv
from mmky import primitives
from mmky.tasks.pour.poursim import PourSim
from mmky.tasks.pour.pourreal import PourReal

GRASP_OFFSET = 0.07

class PourEnv(RomanEnv):
    '''Simplified environment with a reduced action space (x-y motion and wrist rotation)'''
    def __init__(self):
        super().__init__(PourSim, PourReal, os.path.join(os.path.dirname(__file__), 'xyconfig.yaml'))
        self.action_space = Box(low=-1, high=1, shape=(3,))

    def reset(self, **kwargs):
        while True:
            obs = super().reset(**kwargs)
            self.home = obs["arm_state"].tool_pose()
            hx, hy = self.home[:2]
            # grab the source cup
            x, y = obs["world"]["source"]["position"][:2]
            if not primitives.pivot_xy(self.robot, self.home, x, y, 0):
                continue
            if not primitives.pick(self.robot, self.workspace_height + GRASP_OFFSET):
                continue
            if not primitives.pivot_xy(self.robot, self.home, hx, hy, 0):
                continue
            return self._observe()

    def _act(self, action):
        dx, dy, dr = action
        primitives.pivot_dxdy(self.robot, self.home, dx, dy, dr)
        return False # force_full_obs
