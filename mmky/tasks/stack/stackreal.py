from mmky import RealScene

class StackReal(RealScene):
    def reset(self, **kwargs):
        # TODO move the cubes to the new poses. unstack if needed.
        ws = super().reset(**kwargs)
        self.cube_count = len(ws)

    def eval_state(self, world_state):
        rew = self.cube_count - len(world_state)
        success = rew == self.cube_count - 1
        done = success
        return rew, success, done
