import math
import numpy as np
import os
from mmky import SimScene
from mmky import primitives

class PushSim(SimScene):
    def __init__(self, robot, obs_res, workspace, ball_count=2, ball_radius=0.035, cameras={}, **kwargs):
        super().__init__(robot, obs_res, workspace, cameras, **kwargs)
        self.ball_count = ball_count
        self.ball_radius = ball_radius

    def reset(self, **kwargs):
        super().reset(**kwargs)
        colors = [(1, 1, 0, 1), (1, 0.5, 0, 1)]
        tags = ["yellow", "orange"]
        for i in range(self.ball_count):
            ball_pose = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius) + [self.workspace_height]
            self.make_ball(self.ball_radius, ball_pose, color=colors[i], mass=0.04, tag=tags[i])

        self.start_state = self.get_world_state()

    def eval_state(self, world_state):
        rew = 0
        success = False
        done = False
        if np.linalg.norm(self.start_state["orange"]["position"][:2] - world_state["orange"]["position"][:2]) > 0.05 and \
           np.linalg.norm(self.start_state["yellow"]["position"][:2] - world_state["yellow"]["position"][:2]) > 0.05:
            rew = 1
            success = True
            done = True
        return rew, success, done
