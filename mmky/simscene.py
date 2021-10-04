import os
import roman

class SimScene(roman.SimScene):
    def __init__(self, robot, obs_res, cameras={}):
        data_dir = os.path.join(os.path.dirname(__file__), 'sim\\data')
        tex_dir = os.path.join(data_dir, "img")
        super().__init__(robot=robot, data_dir=data_dir, tex_dir=tex_dir)
        self.obs_res = obs_res
        self.cameras = cameras

    def reset(self):
        super().reset()
        for cam_id, cam_def in self.cameras.items():
            self.create_camera(img_res=self.obs_res, tag=cam_id, **cam_def)
