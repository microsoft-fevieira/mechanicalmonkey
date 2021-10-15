import numpy as np
from mmky.env import RomanEnv, RoboSuiteEnv
from mmky.writers import SimpleHdf5Writer
from roman import ur, rq

class Expert:
    def __init__(self, simscenefn, realscenefn, config):
        self.env = RomanEnv(simscenefn, realscenefn, config, log_writer=self)
        self.writer = SimpleHdf5Writer("robosuite_stack")
        self.robot = self.env.robot
        self.images = None
        self.done = False

    def _start_episode(self):
        obs = self.env.reset()
        self.writer.start_episode(RoboSuiteEnv.make_observation(obs))
        self.world = self.env._get_world_state()
        self.done = False

    def _end_episode(self):
        self.done = True
        self.robot.step()
        self.writer.end_episode()

    def __call__(self, *proprio):
        if self.images:
            arm_state, hand_state, arm_cmd, hand_cmd = proprio
            if arm_cmd.kind() == ur.UR_CMD_KIND_MOVE_JOINT_SPEEDS:
                arm_act = arm_cmd.target()
            else:
                arm_act = arm_state.joint_speeds()

            hand_act = 0
            if hand_cmd.kind() == rq.Command._CMD_KIND_MOVE:
                hand_act = 2 * (hand_cmd.position() / 255 - 0.5)

            act = np.zeros(7)
            act[:6] = arm_act
            act[6] = hand_act
            self.world = self.env._get_world_state(False)
            obs = RomanEnv.make_observation(self.images, self.world, proprio)
            rew, _, success = self.env._eval_state(obs)
            obs = RoboSuiteEnv.make_observation(obs)
            self.writer.log(act, obs, rew, self.done, {"success": success})
        self.images = self.env._get_camera_images()

