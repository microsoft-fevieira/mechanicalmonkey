import random
import mmky
from mmky.tasks.stack.stacksim import StackSim
from mmky.tasks.stack.stackreal import StackReal



if __name__ == '__main__':
    env = mmky.env.RoboSuiteEnv(StackSim, StackReal, config="C:\\code\\mechanicalmonkey\\test\\test_config.yaml")
    env.reset()

    for i in range(1000):
        env.step([random.random()-0.5,0,0,0,0,0,0])

