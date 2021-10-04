from mmky import RealScene

class PourReal(RealScene):
    def reset(self, source_cup_pos, target_cup_pos):
        # TODO move the cups to the new poses, replace the ball in a cup if needed
        super().reset()
