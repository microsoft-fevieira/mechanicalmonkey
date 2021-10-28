import numpy as np
from mmky import RealScene
from mmky import primitives

def _is_orange(color):
    return color[0] < 0.5 * color[1] < color[2]

class PourReal(RealScene):
    def __init__(self,
                 robot,
                 obs_res,
                 cameras,
                 cup_size,
                 ball_count,
                 workspace,
                 out_position=None,
                 neutral_position=None,
                 detector=None,
                 **kwargs):
        super().__init__(robot, obs_res, cameras, workspace, out_position, neutral_position, detector)
        self.cup_size = cup_size
        self.ball_count = ball_count

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._source_cup = None
        ws = self.get_world_state(False)
        # TODO move the cups to the new poses, replace the ball in a cup if needed

    def move_home(self, home_pose):
        x, y = home_pose[:2]
        primitives.pivot_xy(self.robot, x, y, 0, reference_pose=home_pose)

    def get_world_state(self, force_state_refresh):
        raw_state = super().get_world_state(force_state_refresh)
        world, balls, cups, unknown = {}, [], [], []
        src_id = -1
        for k,v in raw_state.items():
            if v["size"][0] > 0.05:
                # this is a cup
                # reset the center of the cup to workspace level
                v["position"][2] = self.workspace_height
                v["size"] = np.array(self.cup_size) # the size provided by the detector is quite off (particularly the height)
                v["has_balls"] = _is_orange(v["color"])
                cups.append(v)
            else:
                if _is_orange(v["color"]) and v["size"][0] < 0.05: # small orange object, so its a ball
                    balls.append(v)
                else:
                    # ???
                    unknown.append(v)

        assert len(cups) == 2
        # if any balls are in cups, remove them from the ball set
        spilled_balls = []
        for b in balls:
            in_cup = False
            for c in cups:
                in_cup = in_cup or np.linalg.norm(b["position"][:2] - c["position"][:2]) < c["size"][0] / 2
            if not in_cup:
                spilled_balls.append(b)

        world["unknown"] = unknown
        world["cups"] = cups
        world["balls"] = spilled_balls
        balls = spilled_balls

        if self._source_cup:
            # find the closest detected cup
            pos = self._source_cup["position"][:2]
            src_id = np.argmin([np.linalg.norm(pos - cups[0]["position"][:2]), np.linalg.norm(pos - cups[1]["position"][:2])])
        else:
            # the one with balls is the source cup
            assert cups[0]["has_balls"] or cups[1]["has_balls"]
            src_id = np.argmax([cups[0]["has_balls"], cups[1]["has_balls"]])

        self._source_cup = cups[src_id]
        world["source"] = cups[src_id]
        world["target"] = cups[1 - src_id]

        remaining = 0 if not cups[src_id]["has_balls"] else 1 if (cups[1 - src_id]["has_balls"] or len(balls)) else 2
        spilled = len(balls)
        poured = self.ball_count - remaining - spilled

        world["ball_data"] = {"poured": poured, "remaining": remaining, "spilled": spilled}
        return world

    def eval_state(self, world_state):
        rew = world_state["ball_data"]["poured"]
        success = rew == self.ball_count
        done = world_state["ball_data"]["remaining"] == 0
        return rew, success, done
