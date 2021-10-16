from mmky.tasks.drop.dropreal import DropReal
from mmky.tasks.drop.dropsim import DropSim
from mmky import primitives
from mmky.expert import Expert
import random
import os

GRASP_HEIGHT = 0.04
MAX_ACC = 0.5
MAX_SPEED = 0.5

class DroppingExpert(Expert):
    def __init__(self):
        super().__init__("robosuite_drop", DropSim, DropReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def run(self, iterations=1, data_dir="trajectories"):
        while iterations:
            self._start_episode()

            # move over the object
            target = self.robot.tool_pose
            target[:2] = self.world["obj"]["position"][:2]
            if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # pick the object
            if not primitives.pick(self.robot, self.env.workspace_height + GRASP_HEIGHT, grasp_state=128, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # move over the cup
            target = self.robot.tool_pose
            target[:2] = self.world["cup"]["position"][:2]
            if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # drop the object
            self.robot.release(timeout=2)

            # discard failed tries 
            if not self.success:
                continue
            self._end_episode()
            iterations -= 1

if __name__ == '__main__':
    exp = DroppingExpert()
    exp.run(1000)
