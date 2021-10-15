from mmky.tasks.stack.stackreal import StackReal
from mmky.tasks.stack.stacksim import StackSim
from mmky import primitives
from mmky.expert import Expert
import random
import os

GRASP_HEIGHT = 0.04
MAX_ACC = 0.5
MAX_SPEED = 0.5

class StackingExpert(Expert):
    def __init__(self):
        super().__init__(StackSim, StackReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def stack(self, iterations=1, data_dir="trajectories"):
        while iterations:
            self._start_episode()

            # pick a random cube as the target
            target_id, source_id = random.sample(self.world.keys(), k=2)

            # move over the cube
            target = self.robot.tool_pose
            target[:2] = self.world[source_id]["position"][:2]
            if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # pick the cube
            if not primitives.pick(self.robot, self.env.workspace_height + GRASP_HEIGHT, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # move over the second cube
            target = self.robot.tool_pose
            target[:2] = self.world[target_id]["position"][:2]
            if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # place the cube
            if not primitives.place(self.robot, self.env.workspace_height + GRASP_HEIGHT, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            self._end_episode()


if __name__ == '__main__':
    exp = StackingExpert()
    exp.stack(1000)
