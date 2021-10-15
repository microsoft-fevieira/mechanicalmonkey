from mmky.tasks.stack.stackenv import StackEnv
import random

class XYStackExpert:
    def __init__(self, env: StackEnv):
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

    def run(self, iterations=1):
        while iterations:
            obs = self.env.reset()
            objects = obs["world"]
            # pick a random cube as the target
            target_id, source_id = random.sample(obs["world"].keys(), k=2)

            # get the cube
            home_pose = obs["arm_state"].tool_pose()
            current = home_pose[:2]
            source = objects[source_id]["position"][:2]
            self.move(current, source)
            obs, _, _, _ = self.env.step([0, 0, 1]) # pick

            # place at target location
            current = obs["arm_state"].tool_pose()[:2]
            target = objects[target_id]["position"][:2]
            self.move(current, target)

            obs, rew, _, _ = self.env.step([0, 0, -1]) # place
            print(rew)
            iterations -= 1

if __name__ == '__main__':
    env = StackEnv()
    expert = XYStackExpert(env)
    expert.run(10)
