from mmky.tasks.pour.poursim import PourSim
from mmky.tasks.pour.pourreal import PourReal
from mmky.expert import Expert
from mmky import primitives
import math
import numpy as np
import random
import os

GRASP_HEIGHT = 0.07
MAX_ACC = 0.25
MAX_SPEED = 0.25

class PourExpert(Expert):
    def __init__(self):
        super().__init__("robosuite_pour", PourSim, PourReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def run(self, iterations=1):
        while iterations:
            self._start_episode()
            home_pose = self.robot.tool_pose

            # move over first cup
            first = self.robot.tool_pose
            sx, sy = self.world["source"]["position"][:2]
            if not primitives.pivot_xy(self.robot, sx, sy, 0, reference_pose=home_pose, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # pick the first cup
            if not primitives.pick(self.robot, self.env.workspace_height + GRASP_HEIGHT, 0, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # move to the second cup
            tx, ty = self.world["target"]["position"][:2]
            sign = 1
            if math.atan2(sy, sx) < math.atan2(ty, tx):
                # counter-clockwise rotation in the third quadrant
                sign = -1
            dist = self.world["source"]["size"][0] / 2 + self.world["target"]["size"][0]
            tx, ty = primitives.add_cylindrical(tx, ty, 0, dist * sign) # assume the arc is about the same length as the chord (dist)

            if not primitives.pivot_xy(self.robot, tx, ty, 0, reference_pose=home_pose, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # pour
            before_pour = self.robot.joint_positions  
            target = before_pour + [0, 0, 0, 0, 0, sign * 3 * math.pi / 5]
            self.robot.move(target, max_speed=MAX_SPEED, max_acc=MAX_ACC, timeout=15)
            
            # pick a random spot and set the cup there
            self.robot.move(before_pour, max_speed=MAX_SPEED, max_acc=MAX_ACC, timeout=15)
            x, y = tx, ty = self.world["target"]["position"][:2]
            while np.linalg.norm([x-tx, y-ty]) < self.world["source"]["size"][0] * 2.5:
                x, y = primitives.generate_random_xy(*self.env.workspace_span, *self.env.workspace_radius)
            primitives.pivot_xy(self.robot, x, y, 0, reference_pose=home_pose, max_speed=MAX_SPEED, max_acc=MAX_ACC)
            primitives.place(self.robot, self.env.workspace_height + GRASP_HEIGHT, max_speed=MAX_SPEED, max_acc=MAX_ACC)

            # check what happened
            self._writer_enabled = False
            self.env.scene.get_world_state(force_state_refresh=True)
            self._writer_enabled = True

            # make sure we get another observation
            self.robot.stop()

            # discard failed tries
            if not self.success:
                print(f"Episode skipped. State evaluated as: {self.env.scene.eval_state(self.world)}")
                input(f"Fix the scene and press enter to continue")
                continue
            self._end_episode()
            iterations -= 1


if __name__ == '__main__':
    expert = PourExpert()
    expert.run(1000)
