from mmky import RealScene

class PourReal(RealScene):
    def reset(self, cubePoses):
        # TODO move the cups to the new poses, replace the ball in a cup if needed
        super().reset()
