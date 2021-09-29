import roman 

class SimScene(roman.SimScene):
    def __init__(self, robot, obs_res, cameras={}):
        super().__init__(robot, None)
        for cam_id, cam_def in cameras.items():
            self.create_camera(img_res=obs_res, tag=cam_id, **cam_def)


