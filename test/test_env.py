import random
import mmky
from mmky.tasks.stack.stacksim import StackSim
from mmky.tasks.stack.stackreal import StackReal
import os

cfg_file = os.path.join(os.path.dirname(__file__), 'test_config.yaml')

def test_step():
    env = mmky.env.RoboSuiteEnv(StackSim, StackReal, config=cfg_file)
    env.reset()

    for i in range(100):
        env.step([random.random()-0.5,0,0,0,0,0,0])

    env.close()

def test_close():
    for s in range(5):
        env = mmky.env.RoboSuiteEnv(StackSim, StackReal, config=cfg_file)
        env.reset()
        for i in range(100):
            env.step([random.random()-0.5,0,0,0,0,0,0])
        env.close()
        print(s)

if __name__ == '__main__':
    test_step()
    test_close()