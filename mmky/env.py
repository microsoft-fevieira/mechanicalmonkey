import gym
from gym.spaces import Box, Dict, Tuple
import numpy as np
import random
import torch
from roman import Robot, GraspMode, Joints, Tool, JointSpeeds
from roman.ur import arm
from roman.rq import hand
from mmky.realscene import RealScene
from mmky.simscene import SimScene
from mmky import primitives
import cv2
import yaml


MAX_OBJECTS_IN_SCENE = 10

class RomanEnv(gym.Env):
    def __init__(self, simscenefn=SimScene, realscenefn=RealScene, config={}, log_writer=None):
        super().__init__()
        if type(config) is str:
            with open(config) as f:
                config = yaml.safe_load(f)
        self.config = config
        use_sim = config.get("use_sim", True)
        robot_config = config.get("robot", {})
        self.robot = Robot(use_sim=use_sim, config=robot_config, writer=log_writer)
        self.obs_res = config.get("obs_res", (84, 84))
        ws = config.get("workspace", {"radius": [3.5, 4.2], "span": [0.5, 0.75], "height": 0})
        self.workspace_radius, self.workspace_span, self.workspace_height = ws.values()
        scene_fn, scene_cfg = (simscenefn, "sim_scene") if use_sim else (realscenefn, "real_scene")
        scene_config = config.get(scene_cfg, {})
        self.scene = scene_fn(robot=self.robot, obs_res=self.obs_res, workspace=ws, **scene_config)
        self.robot.connect()
        self.scene.connect()
        self.render_mode = config.get("render_mode", None)
        self.max_steps = config.get("max_steps", -1)
        self.grasp_mode = config.get("grasp_mode", None)
        self.grasp_mode = eval(self.grasp_mode) if self.grasp_mode else GraspMode.BASIC
        self.random_start = config.get("random_start", True)

        camera_count = self.scene.get_camera_count()
        self.observation_space = Dict({
            "cameras": Tuple(camera_count * [Box(low=0, high=255, shape=(self.obs_res[0], self.obs_res[1], 3), dtype=np.uint8)]),
            "world": Box(low=-2, high=2, shape=(MAX_OBJECTS_IN_SCENE,)),
            "arm_state": Box(low=-np.inf, high=np.inf, shape=(arm.State._BUFFER_SIZE,)),
            "hand_state": Box(low=-np.inf, high=np.inf, shape=(hand.State._BUFFER_SIZE,)),
            "last_arm_cmd": Box(low=-np.inf, high=np.inf, shape=(arm.Command._BUFFER_SIZE,)),
            "last_hand_cmd": Box(low=-np.inf, high=np.inf, shape=(hand.Command._BUFFER_SIZE,))})

    def close(self):
        self.scene.disconnect()
        self.scene = None
        self.robot.disconnect()
        self.robot = None

    def seed(seed=None):
        """Sets the seed for this env's random number generator."""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self, **kwargs):
        self.step_count = 0
        self.total_reward = 0
        self.is_done = False
        self.success = False
        self.scene.reset(**kwargs)
        self.start_pose = self.robot.joint_positions
        primitives.go_to_start(self.robot, self.start_pose, self.grasp_mode)
        tool_pose = self.robot.tool_pose
        if self.random_start:
            tool_pose[:2] = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius)
        self.robot.move(tool_pose, max_speed=1, max_acc=1)
        return self._observe()

    def end_episode(self, success=True):
        self.is_done = True
        self.success = success
        #self.scene.end_episode()

    def step(self, action):
        self.info = {}
        force_state_refresh = self._act(action)
        obs = self._observe(force_state_refresh)
        rew, success, done = self._eval_state(obs)
        done = done or self.is_done or self.step_count >= self.max_steps
        self.info["success"] = self.success or success
        self.step_count += 1
        self.info["step_count"] = self.step_count
        self.total_reward += rew
        self.info["total_reward"] = self.total_reward
        self.render(self.render_mode)
        return obs, rew, done, self.info

    def render(self, mode='human'):
        img = self._last_state["cameras"][0]
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow("camera observation", img)
            cv2.waitKey(1)

    def _observe(self, force_state_refresh=False):
        images = self._get_camera_images()
        world = self._get_world_state(force_state_refresh)
        proprio = self._get_proprioceptive_state()
        self._last_state = self.make_observation(images, world, proprio)
        return self._last_state

    @staticmethod
    def make_observation(images, world, proprio):
        arm_state, hand_state, arm_cmd, state_cmd = proprio
        return {
            "cameras": images,
            "world": world,
            "arm_state": arm_state,
            "hand_state": hand_state,
            "last_arm_cmd": arm_cmd,
            "last_hand_cmd": state_cmd}

    def _get_camera_images(self):
        return self.scene.get_camera_images() # this is blocking

    def _get_world_state(self, force_state_refresh=False):
        return self.scene.get_world_state(force_state_refresh)

    def _get_proprioceptive_state(self):
        self.robot.step() # update the proprioceptive state without interrupting the motion or overriding the commands
        arm_state, hand_state = self.robot.last_state()
        arm_cmd, hand_cmd = self.robot.last_command()
        return arm_state, hand_state, arm_cmd, hand_cmd

    def _act(self, action):
        # must be provided by derived classes or externalized to an expert
        return False # don't force_state_refresh

    def _eval_state(self, obs):
        return self.scene.eval_state(obs["world"])


class RoboSuiteEnv(RomanEnv):
    def __init__(self, simscenefn=SimScene, realscenefn=RealScene, config={}):
        super().__init__(simscenefn=simscenefn, realscenefn=realscenefn, config=config)
        self.action_space = Box(low=-1, high=1, shape=(7,))
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(self.obs_res[0], self.obs_res[1], 3), dtype=np.uint8),
            "proprio": Box(low=-np.inf, high=np.inf, shape=(37,))})

    def reset(self, **kwargs):
        self.__last_hand_target = 1
        return super.reset(**kwargs)

    def _act(self, action):
        self.env.robot.move(JointSpeeds(*action[:6]), max_speed=1, max_acc=0.5, timeout=0)
        if action[6] != self.__last_hand_target:
            self.__last_hand_target = action[6]
            pos = min(255, max(action[6] * 255, 0))
            self.env.robot.grasp(position=pos, timeout=0)
        return False

    def _observe(self):
        obs = super()._observe()
        return self.convert_observation(obs)

    @staticmethod
    def make_observation(obs):
        joint_pos_cos = np.cos(obs["arm_state"].joint_positions())
        joint_pos_sin = np.sin(obs["arm_state"].joint_positions())
        joint_vel = np.sin(obs["arm_state"].joint_speeds())
        eef = obs["arm_state"].tool_pose()
        eef_pos = eef.position()
        eef_quat = eef.orientation()
        gripper_qpos = [obs["hand_state"].position_A(), 0, obs["hand_state"].position_B(), 0, obs["hand_state"].position_C(), 0]
        gripper_qvel = [0, 0, 0, 0, 0, 0]
        return {"image": obs["cameras"][0],
                "proprio": np.concatenate((joint_pos_cos,
                                          joint_pos_sin,
                                          joint_vel,
                                          eef_pos,
                                          eef_quat,
                                          gripper_qpos,
                                          gripper_qvel))}

