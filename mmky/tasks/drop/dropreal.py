from mmky import RealScene
from mmky import primitives
import random
import numpy as np

GRASP_DEPTH = 0.02

######################################
# Assumes a bottomless container/cup
######################################
class DropReal(RealScene):
    def __init__(self,
                 robot,
                 obs_res,
                 cameras,
                 cup_size,
                 obj_count,
                 workspace,
                 out_position=None,
                 neutral_position=None,
                 detector=None, 
                 **kwargs):
        super().__init__(robot, obs_res, cameras, workspace, out_position, neutral_position, detector)
        self.cup_size = cup_size
        self.obj_count = obj_count

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._source_cup = None
        ws = self.get_world_state(False)

        if len(ws["obj"]) == self.obj_count:
            return

        target = self.robot.tool_pose
        target[:2] = ws["cup"]["position"][:2]
        self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
        
        # pick the bottomless cup - the objects inside should be left behind
        primitives.pick(self.robot, self.workspace_height + self.cup_size[2] - GRASP_DEPTH, pre_grasp_size=0, max_speed=0.5, max_acc=0.5)

        # pick a random spot and drop the object there
        x, y = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius)
        target[:2] = x, y
        self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
        primitives.place(self.robot, self.workspace_height + self.cup_size[2] - GRASP_DEPTH, max_speed=0.5, max_acc=0.5)
        ws = self.get_world_state(force_state_refresh=True)

    def get_world_state(self, force_state_refresh):
        raw_state = super().get_world_state(force_state_refresh)
        world, obj, cup, unknown = {}, [], None, []
        src_id = -1

        sizes = np.array([np.sum(v["size"]) for v in raw_state.values()])
        cup_id = np.argmax(sizes)
        cup = raw_state[cup_id].copy()
        cup["position"][2] = self.workspace_height
        cup["size"] = np.array(self.cup_size) # the size provided by the detector is quite off (particularly the height)

        # if any objects are in cup, remove them from the object set
        out_obj = []
        obj = list([raw_state[i] for i in range(len(raw_state)) if i != cup_id])
        for o in obj:
            if np.linalg.norm(o["position"][:2] - cup["position"][:2]) > cup["size"][0] / 2:
                out_obj.append(o)

        world["cup"] = cup
        world["obj"] = out_obj

        return world

    def eval_state(self, world_state):
        rew = self.obj_count - len(world_state["obj"])
        success = rew == self.obj_count
        done = success
        return rew, success, done
