from mmky import RealScene

class PushReal(RealScene):
    def reset(self, **kwargs):
        self.start_world = super().reset(**kwargs)

    def eval_state(self, world_state):
        raise NotImplementedError()
        #return rew, success, done
