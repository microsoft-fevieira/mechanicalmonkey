from mmky import RealScene

class PlaceReal(RealScene):
    def reset(self, **kwargs):
        ws = super().reset(**kwargs)

    def eval_state(self, world_state):
        rew = 2 - len(world_state)
        success = rew == 1
        done = success
        return rew, success, done
