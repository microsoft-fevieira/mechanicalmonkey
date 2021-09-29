import mmky
from mmky.tasks.stack.stackenv import StackEnv

if __name__ == '__main__':
    env = StackEnv()
    env.render()
    env.reset()
    while(True):
        env.render()
        env.step([0, 0 ,0])
