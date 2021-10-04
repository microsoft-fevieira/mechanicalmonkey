from mmky.tasks.pour.pourenv import PourEnv
import random

class PourExpert:
    def __init__(self, env: PourEnv):
        self.env = env

    def move(self, current, target):
        x = 1
        y = 0
        while x or y:
            delta = target - current
            x = 0 if abs(delta[0]) < 0.002 else 1 if delta[0] > 0 else -1
            y = 0 if abs(delta[1]) < 0.002 else 1 if delta[1] > 0 else -1
            obs, _, _, _ = self.env.step([x, y, 0])
            current = obs["arm_state"].tool_pose()[:2]
        for i in range(10):
            self.env.step([0, 0, 0])

    def pour(self):
        obs = self.env.reset()
        objects = obs["world"]
        home_pose = obs["arm_state"].tool_pose()

        
        

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
    env = PourEnv()
    expert = PourExpert(env)
    expert.pour()
