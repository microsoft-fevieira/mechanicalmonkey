import random
import mmky
from mmky.tasks.stack.stacksim import StackSim
from mmky.tasks.stack.stackreal import StackReal
import os


if __name__ == '__main__':
    cfg_file = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    env = mmky.env.RoboSuiteEnv(StackSim, StackReal, config=cfg_file)
    env.reset()

    for i in range(100):
        env.step([random.random()-0.5,0,0,0,0,0,0])

