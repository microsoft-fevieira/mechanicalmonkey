from cv2 import sepFilter2D
import os
import numpy as np
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
            self.target_pos = obs["world"]["target"]["position"][:2]
            self.target_size = obs["world"]["target"]["size"][0]
            hx, hy = self.home[:2]
            # grab the source cup
            x, y = obs["world"]["source"]["position"][:2]
            if not primitives.pivot_xy(self.robot, x, y, 0, reference_pose=self.home):
                continue
            self._src_cup_pos = self.robot.joint_positions
            if not primitives.pick(self.robot, self.workspace_height + GRASP_OFFSET, pre_grasp_size=0):
                continue
            if not primitives.pivot_xy(self.robot, hx, hy, 0, reference_pose=self.home):
                continue
            return self._observe()

    def finalize(self):
        self.robot.move(self._src_cup_pos, max_speed=0.5, max_acc=0.5)
        # pick a random spot and set the cup there
        x, y = tx, ty = self.target_pos
        while np.linalg.norm([x-tx, y-ty]) < self.target_size * 2.5:
            x, y = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius)
        primitives.pivot_xy(self.robot, x, y, 0, reference_pose=self.home, max_speed=0.5, max_acc=0.5)
        primitives.place(self.robot, self.workspace_height + GRASP_OFFSET, max_speed=0.3, max_acc=0.3)
        self.scene.get_world_state(force_state_refresh=True)
        return self.step([0, 0, 0]) 

    def _act(self, action):
        dx, dy, dr = np.array(action, dtype=int)
        primitives.pivot_dxdy(self.robot, dx, dy, dr, reference_pose=self.home, max_speed=0.3, max_acc=0.3)
        return False # force_full_obs
