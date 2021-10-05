from mmky import SimScene

class StackSim(SimScene):
    def __init__(self, robot, obs_res, workspace_height=0, cube_size=0.05, cameras={}):
        super().__init__(robot, obs_res, workspace_height, cameras)
        self.cube_size = cube_size

    def reset(self, cube_poses):
        super().reset()
        size = [self.cube_size] * 3
        for i in range(len(cube_poses)):
            self.make_box(size, cube_poses[i], color=(0.8, 0.2, 0.2, 1), mass=0.1, tag=i)

