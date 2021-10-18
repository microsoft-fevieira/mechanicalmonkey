from mmky.tasks.push.pushreal import PushReal
from mmky.tasks.push.pushsim import PushSim
from mmky import primitives
from mmky.expert import Expert
import random
import os

GRASP_HEIGHT = 0.04
MAX_ACC = 0.5
MAX_SPEED = 0.5

class PushingExpert(Expert):
    def __init__(self):
        super().__init__("robosuite_push", PushSim, PushReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def run(self, iterations=1, data_dir="trajectories"):
        while iterations:
            self._start_episode()

            # pick a ball
            cue_ball, target_ball = random.sample(self.world.keys(), k=2)
            pose = self.robot.tool_pose
            pose[:2] = self.world[cue_ball]["position"][:2]
            if not self.robot.move(pose, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # discard failed tries 
            if not self.success:
                continue
            self._end_episode()
            iterations -= 1

if __name__ == '__main__':
    exp = PushingExpert()
    exp.run(1000)
