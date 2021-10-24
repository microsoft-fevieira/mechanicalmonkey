from mmky.tasks.drop.dropreal import DropReal
from mmky.tasks.drop.dropsim import DropSim
from mmky import primitives
from mmky.expert import Expert
import random
import os

OBJ_GRASP_HEIGHT = 0.02
CUP_GRASP_HEIGHT = 0.08
MAX_ACC = 0.25
MAX_SPEED = 0.25

class DroppingExpert(Expert):
    def __init__(self):
        super().__init__("robosuite_drop", DropSim, DropReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def run(self, iterations=1, data_dir="trajectories"):
        while iterations:
            self._start_episode()
            while len(self.world["obj"]):
                # move over an object
                target = self.robot.tool_pose
                tobj = random.choice(self.world["obj"])
                target[:2] = tobj["position"][:2]
                if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                    continue

                # pick the object
                if not primitives.pick(self.robot, self.env.workspace_height + OBJ_GRASP_HEIGHT, pre_grasp_size=60, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                    continue

                # move over the cup
                target = self.robot.tool_pose
                target[:2] = self.world["cup"]["position"][:2]
                if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                    continue

                # drop the object
                self.robot.release(timeout=2)

                # check what happened
                self._writer_enabled = False
                back = self.robot.tool_pose
                self.env.scene.get_world_state(force_state_refresh=True)
                self.robot.move(back, timeout=10, max_speed=0.5, max_acc=0.5)
                self._writer_enabled = True

                # make sure we get another observation
                self.robot.stop()

            # discard failed tries
            if not self.success:
                continue
            self._end_episode()
            iterations -= 1

if __name__ == '__main__':
    exp = DroppingExpert()
    exp.run(100)
