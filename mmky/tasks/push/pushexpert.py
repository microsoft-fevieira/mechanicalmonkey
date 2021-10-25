from mmky.tasks.push.pushreal import PushReal
from mmky.tasks.push.pushsim import PushSim
from mmky import primitives
from mmky.expert import Expert
from roman import Joints, Tool
import random
import os
import math
import numpy as np

PUSH_DEPTH = 0.03
MAX_ACC = 0.25
MAX_SPEED = 0.25

class PushingExpert(Expert):
    def __init__(self):
        super().__init__("robosuite_push", PushSim, PushReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def run(self, iterations=1, data_dir="trajectories"):
        while iterations:
            self._start_episode()

            o_pos = self.world["object"]["position"][:2]
            o_z = self.world["object"]["position"][2]
            o_size = self.world["object"]["size"]
            t_pos = self.world["target"]["position"][:2]
            #t_pos = np.array([-0.461, -0.377]) # center
            d = (t_pos - o_pos) / np.linalg.norm(t_pos - o_pos)
            theta = math.atan2(t_pos[1] - o_pos[1], t_pos[0] - o_pos[0])

            # pick a point on the same line, offset by half the size of the ball
            x, y = o_pos + d * -o_size[0]
            base_rotation = self.robot.joint_positions[Joints.BASE]
            if not primitives.move_xy(self.robot, x, y, theta, max_speed=MAX_SPEED, max_acc=MAX_ACC, max_time=20):
                continue

            # lower the arm
            high_z = self.robot.tool_pose[Tool.Z]  
            low = self.robot.tool_pose
            low[Tool.Z] = o_z - PUSH_DEPTH
            if not self.robot.move(low, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # move towards the center
            x, y = t_pos + d * -o_size[0] / 2
            if not primitives.move_xy(self.robot, x, y, theta, max_speed=MAX_SPEED, max_acc=MAX_ACC, max_time=20):
                continue

            # move back up and check result
            high = self.robot.tool_pose
            high[Tool.Z] = high_z
            self.robot.move(high)
            self._writer_enabled = False
            self.env.scene.get_world_state(force_state_refresh=True)
            self._writer_enabled = True

            # make sure we get another observation
            self.robot.stop()

            # discard failed tries 
            if not self.success:
                continue
            self._end_episode()
            iterations -= 1

if __name__ == '__main__':
    exp = PushingExpert()
    exp.run(1000)
