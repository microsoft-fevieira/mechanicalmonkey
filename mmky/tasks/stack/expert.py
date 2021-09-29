from mmky.tasks.stack.stackenv import StackEnv
from roman import Robot, Tool
import random
import numpy as np

class StackExpert:
    def __init__(self, env: StackEnv):
        self.env = env

    def move(self, current, target):
        while not np.allclose(current, target, rtol=0, atol=0.01):
            delta = target - current
            x = 0 if abs(delta[0]) < 0.002 else 1 if delta[0] > 0 else -1
            y = 0 if abs(delta[1]) < 0.002 else 1 if delta[1] > 0 else -1
            obs, _, _, _ = self.env.step([x, y, 0])
            current = obs["arm_state"].tool_pose()[:2]

    def stack(self):
        obs = self.env.reset()
        objects = obs["world"]
        home_pose = obs["arm_state"].tool_pose()

        # pick a random cube as the target
        target_id, source_id = random.sample(objects.keys(), k=2)

        # get the cube
        current = home_pose[:2]
        source = objects[source_id][:2]
        self.move(current, source)

        obs, _, _, _ = self.env.step([0, 0, 1]) # pick

        # place at target location
        current = obs["arm_state"].tool_pose()[:2]
        target = objects[target_id][:2]
        self.move(current, target)

        obs, _, _, _ = self.env.step([0, 0, -1]) # place

        # verfy success
        #verify()

        # reset the scene

        #reset()

if __name__ == '__main__':
    env = StackEnv()
    expert = StackExpert(env)
    env.render()
    expert.stack()