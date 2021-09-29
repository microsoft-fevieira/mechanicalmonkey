from mmky import RealScene

class StackReal(RealScene):
    def reset(self, cubePoses):
        # TODO move the cubes to the new poses. unstack if needed.
        super().reset()
