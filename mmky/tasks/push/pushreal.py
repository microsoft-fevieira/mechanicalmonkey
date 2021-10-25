from mmky import RealScene
from mmky import primitives
import numpy as np

GRASP_DEPTH = 0.03
class PushReal(RealScene):
    def reset(self, **kwargs):
        ws = self.get_world_state(False)
        while self.eval_state(ws)[1]:
            primitives.reach_and_pick(self.robot, ws["object"]["position"], -GRASP_DEPTH, max_speed=0.5, max_acc=0.5)
            x, y = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius)
            target = self.robot.tool_pose
            target[:2] = x, y
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
            self.robot.release(0)
            super().reset(**kwargs)
            ws = self.get_world_state(False)

    def get_world_state(self, force_state_refresh):
        raw_state = super().get_world_state(force_state_refresh)
        if not raw_state:
            raw_state = super().get_world_state(True)
        world = {}
        world["object"] = raw_state[0]
        world["target"] = {"position": np.array([-0.461, -0.377])} # center
        return world

    def eval_state(self, world_state):
        rew = 0
        success = False
        done = False
        if np.linalg.norm(world_state["target"]["position"][:2] - world_state["object"]["position"][:2]) < 0.05:
            rew = 1
            success = True
            done = True
        return rew, success, done
