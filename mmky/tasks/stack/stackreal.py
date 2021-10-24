from mmky import RealScene
from mmky import primitives

GRASP_HEIGHT = 0.06
class StackReal(RealScene):
    def reset(self, **kwargs):
        super().reset(**kwargs)
        ws = self.get_world_state(force_state_refresh=False)
        cube_count = len(ws)

        while cube_count == 1:
            self.robot.pinch(60)
            # the cubes are stacked, unstack
            # move over the cube
            target = self.robot.tool_pose
            target[:2] = ws[0]["position"][:2]
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
            
            # pick the cube
            primitives.pick(self.robot, self.workspace_height + GRASP_HEIGHT, max_speed=0.5, max_acc=0.5)

            # pick a random spot and drop the object there
            x, y = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius)
            target[:2] = x, y
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
            self.robot.release(0)
            ws = self.get_world_state(force_state_refresh=True)
            cube_count = len(ws)

        self.cube_count = len(ws)
        assert self.cube_count > 1

    def eval_state(self, world_state):
        rew = self.cube_count - len(world_state)
        success = rew == self.cube_count - 1
        done = success
        return rew, success, done
