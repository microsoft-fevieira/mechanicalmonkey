import mmky
from mmky.tasks.stack.stackenv import StackEnv
from mmky.tasks.stack.expert import StackExpert

if __name__ == '__main__':
    env = StackEnv()
    expert = StackExpert(env)
    env.render()
    expert.stack()
