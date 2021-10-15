from mmky.tasks.pour.poursim import PourSim
from mmky.tasks.pour.pourreal import PourReal
from mmky.expert import Expert
from mmky import primitives
import math
import random
import os

GRASP_HEIGHT = 0.07
MAX_ACC = 0.5
MAX_SPEED = 0.5

class PourExpert(Expert):
    def __init__(self):
        super().__init__("robosuite_pour", PourSim, PourReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def run(self, iterations=1):
        while iterations:
            self._start_episode()
            home_pose = self.robot.tool_pose

            # move over first cup
            first = self.robot.tool_pose
            tx, ty = self.world["source"]["position"][:2]
            tx, ty = primitives.add_cylindrical(tx, ty, 0, 0)
            first[:2] = tx, ty
            if not self.robot.move(first, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # pick the first cup
            if not primitives.pick(self.robot, self.env.workspace_height + GRASP_HEIGHT, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # move to the second cup
            sx, sy = home_pose[:2]
            tx, ty = self.world["target"]["position"][:2]
            sign = 1
            if math.atan2(sy, sx) < math.atan2(ty, tx):
                # counter-clockwise rotation in the third quadrant
                sign = -1
            dist = self.world["source"]["size"][0] / 2 + self.world["target"]["size"][0]
            tx, ty = primitives.add_cylindrical(tx, ty, 0, dist * sign) # assume the arc is about the same length as the chord (dist)

            if not primitives.pivot_xy(self.robot, home_pose, tx, ty, 0, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # pour
            target = self.robot.joint_positions + [0, 0, 0, 0, 0, sign * 3 * math.pi / 5]
            self.robot.move(target, max_speed=MAX_SPEED, max_acc=MAX_ACC, timeout=10)

            # discard failed tries 
            if not self.success:
                continue
            self._end_episode()
            iterations -= 1


if __name__ == '__main__':
    expert = PourExpert()
    expert.run(100)
