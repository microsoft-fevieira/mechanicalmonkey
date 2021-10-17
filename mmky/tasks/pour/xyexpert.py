from mmky import primitives
from mmky import writers
from mmky.tasks.pour.pourenv import PourEnv
import math

class XYPourExpert:
    def __init__(self, env: PourEnv, writer: writers.Writer):
        self.env = env
        self.writer = writer

    def step(self, action):
        res = self.env.step(action)
        self.writer.log(action, *res)
        return res

    def move(self, current, target):
        x = 1
        y = 0
        while x or y:
            delta = target - current
            x = 0 if abs(delta[0]) < 0.002 else 1 if delta[0] > 0 else -1
            y = 0 if abs(delta[1]) < 0.002 else 1 if delta[1] > 0 else -1
            obs, rew, done, info = self.step([x, y, 0])
            current = obs["arm_state"].tool_pose()[:2]
        for i in range(10):
            obs, rew, done, info = self.step([0, 0, 0])

    def run(self, iterations=1):
        while iterations:
            obs = self.env.reset()
            self.writer.start_episode(obs)

            # move to the second cup
            current_xy = obs["arm_state"].tool_pose()[:2]
            hx, hy = current_xy
            tx, ty = obs["world"]["target"]["position"][:2]

            # determine the direction of motion and compute the offset from the target to move the arm to
            sign = -1 if math.atan2(hx, hy) < math.atan2(ty, tx) else 1
            dist = obs["world"]["source"]["size"][0] / 2 + obs["world"]["target"]["size"][0]
            target_xy = primitives.add_cylindrical(tx, ty, 0, dist * sign) # assume the arc is about the same length as the chord (dist)

            # move in discrete increments
            x, y = 1, 1
            while x or y:
                delta = target_xy - current_xy
                x = 0 if abs(delta[0]) < 0.002 else 1 if delta[0] > 0 else -1
                y = 0 if abs(delta[1]) < 0.002 else 1 if delta[1] > 0 else -1
                obs, rew, done, info = self.step([x, y, 0])
                current_xy = obs["arm_state"].tool_pose()[:2]

            # stop
            for i in range(10):
                self.step([0, 0, 0])

            # pour
            target = obs["arm_state"].joint_positions()[5] + sign * 3 * math.pi / 5
            current = obs["arm_state"].joint_positions()[5]
            while (target - current) * sign > 0:
                obs, rew, done, info = self.step([0, 0, sign])
                current = obs["arm_state"].joint_positions()[5]

            # stop
            for i in range(10):
                self.step([0, 0, 0])

            self.writer.end_episode(not info["success"])
            iterations -= 1


if __name__ == '__main__':
    env = PourEnv()
    writer = writers.SimpleNpyWriter("cup_pour_simple")
    expert = XYPourExpert(env, writer)
    expert.run(500)
