from mmky import RealScene

class StackReal(RealScene):
    def reset(self, cubePoses):
        # move the arm out and survey the scene
        self._update_state()

        # TODO move the cubes to the new poses. unstack if needed.
        super().reset()
