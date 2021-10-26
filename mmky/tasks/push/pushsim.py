import math
import numpy as np
import os
from mmky import SimScene
from mmky import primitives

class PushSim(SimScene):
    def __init__(self, robot, obs_res, workspace, ball_radius=0.035, cameras={}, **kwargs):
        super().__init__(robot, obs_res, workspace, cameras, **kwargs)
        self.ball_radius = ball_radius

    def reset(self, **kwargs):
        super().reset(**kwargs)
        ball_pose = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius) + [self.workspace_height]
        self.make_ball(self.ball_radius, ball_pose, color=(1, 0.5, 0, 1), mass=0.04, tag="object", rollingFriction=10)
        target_pose = [-0.461, -0.377, self.workspace_height]
        self.make_box([0.02, 0.02, 0.0001], target_pose, mass=0, tag="target")

        self.start_state = self.get_world_state()

    def eval_state(self, world_state):
        rew = 0
        success = False
        done = False
        if np.linalg.norm(world_state["target"]["position"][:2] - world_state["object"]["position"][:2]) < 0.05:
            rew = 1
            success = True
            done = True
        return rew, success, done
