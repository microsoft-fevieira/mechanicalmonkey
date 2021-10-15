from mmky.tasks.pour.pourenv import PourEnv
import math
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

    def run(self, iterations=1):
        while iterations:
            obs = self.env.reset()
            objects = obs["world"]
            home_pose = obs["arm_state"].tool_pose()

            # move to the second cup
            current = obs["arm_state"].tool_pose()[:2]
            tx, ty = obs["world"]["target"]["position"][:2]

            sign = 1
            if math.atan2(current[1], current[0]) < math.atan2(ty, tx):
                # counter-clockwise rotation in the third quadrant
                sign = -1
            dist = obs["world"]["source"]["size"][0] / 2 + obs["world"]["target"]["size"][0]
            tx, ty = self.env.shift(tx, ty, 0, dist * sign) # assume the arc is about the same length as the chord (dist)
            self.move(current, [tx, ty])

            # pour
            target = obs["arm_state"].joint_positions()[5] + sign * 3 * math.pi / 5
            while abs(obs["arm_state"].joint_positions()[5] - target) > 0.01:
                obs, rew, done, _ = self.env.step([0, 0, sign])
            print(self.env.step_count)
            print(self.env.total_reward)
            # verfy success
            #verify()

            # reset the scene

            #reset()
            iterations -= 1


if __name__ == '__main__':
    env = PourEnv()
    expert = PourExpert(env)
    expert.run(10)
