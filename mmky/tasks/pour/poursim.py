from roman import SimScene

class PourSim(SimScene):
    def __init__(self, robot, obs_res, cup_size=0.10, cameras={}):
        super().__init__(robot, None)
        self.cup_size = cup_size

    def reset(self, cup_poses):
        super().reset()

        #TODO: add cups
