import math
import numpy as np
import os
from mmky import SimScene
from mmky import primitives

class PushSim(SimScene):
    def __init__(self, robot, obs_res, workspace, ball_count=2, ball_radius=0.035, rand_colors=False, cameras={}, **kwargs):
        super().__init__(robot, obs_res, workspace, cameras, **kwargs)
        self.ball_count = ball_count
        self.ball_radius = ball_radius
        self.rand_colors = rand_colors

    def reset(self, **kwargs):
        super().reset(**kwargs)
        
        for i in range(self.ball_count):
            ball_pose = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius) + [self.workspace_height]
            color = [random.random(), random.random(), random.random(), 1] if self.rand_colors else [0.8, 0.2, 0.2, 1]
            self.make_ball(self.ball_radius, ball_pose, color=color, mass=0.04, tag=f"ball{i}")

        self.start_world = self.get_world_state()

    def eval_state(self, world_state):
        rew = 0
        for i in range(self.ball_count):
            tag = f"ball{i}"
            if not np.allclose(world_state[tag]["position"][:2], self.start_world[tag]["position"][:2]):
                rew += 1 

        success = rew == self.ball_count
        done = False # let the agent decide 
        return rew, success, done
