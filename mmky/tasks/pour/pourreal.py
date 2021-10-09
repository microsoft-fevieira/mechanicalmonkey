import numpy as np
from mmky import RealScene

def _is_orange(color):
    return color[0] < 0.5 * color[1] < 0.75 * color[2]

class PourReal(RealScene):
    def __init__(self,
                 robot,
                 obs_res,
                 cameras,
                 cup_size,
                 ball_count,
                 workspace_height=0,
                 out_position=None,
                 neutral_position=None,
                 detector=None):
        super().__init__(robot, obs_res, cameras, workspace_height, out_position, neutral_position, detector)
        self.cup_size = cup_size
        self.ball_count = ball_count

    def reset(self, source_cup_pos, target_cup_pos):
        # TODO move the cups to the new poses, replace the ball in a cup if needed
        return super().reset()

    def get_world_state(self, force_state_refresh):
        raw_state = super().get_world_state(force_state_refresh)
        
        world = {}
        balls = []
        world["balls"] = balls
        unknown = []
        world["unknown"] = unknown
        for k,v in raw_state.items():
            if v["size"][0] > 0.04:
                # this is a cup
                # reset the center of the cup to workspace level
                v["position"][2] = self.workspace_height
                v["size"] = np.array(self.cup_size) # the size provided by the detector is quite off (particularly the height) 

                # if orange, it's a source cup 
                if (_is_orange(v["color"]) and not world.get("source", None)) or world.get("target", None):
                    world["source"] = v
                else:
                    world["target"] = v

            else:
                if _is_orange(v["color"]) and v["size"][0] < 0.04: # small orange object, so its a ball
                    balls.append(v)
                else:
                    # ???
                    unknown.append(v)

        world["ball_data"] = {"poured": 0, "remaining": self.ball_count - len(balls), "spilled": len(balls)}
        return world

