from roman import SimScene

class StackSim(SimScene):
    def __init__(self, robot, obs_res, cube_size=0.05, cameras={}):
        super().__init__(robot, None)
        self.cube_size = cube_size
        for cam_id, cam_def in cameras.items():
            self.create_camera(img_res=obs_res, tag=cam_id, **cam_def)

    def reset(self, cube_poses):
        super().reset()
        size = [self.cube_size] * 3
        for i in range(len(cube_poses)):
            self.make_box(size, cube_poses[i], color=(0.8, 0.2, 0.2, 1), mass=0.1, tag=i)

