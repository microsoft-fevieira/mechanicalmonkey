import math
import cv2
from roman import Robot, Joints, Tool
from mmky import k4a
from mmky.detector import KinectDetector
HALF_PI = math.pi / 2

class RealScene:
    def __init__(self,
                 robot: Robot,
                 detector_cam_id,
                 obs_res,
                 out_position,
                 neutral_position,
                 cameras):
        self.robot = robot
        self.detector_cam_id = detector_cam_id
        self.obs_res = obs_res
        self.out_position = eval(out_position)
        self.neutral_position = eval(neutral_position)
        self.detector = KinectDetector(id=detector_cam_id)
        self.cameras = {}
        for cam_tag, cam_def in cameras.items():
            if cam_def["type"] == "k4a":
                self.cameras[cam_tag] = k4a.Device.open(cam_def["id"])
            else:
                raise ValueError(f'Unsupported camera type {cam_def["type"]}. ')

        self.k4a_config = k4a.DeviceConfiguration(color_format=k4a.EImageFormat.COLOR_BGRA32, depth_mode=k4a.EDepthMode.NFOV_UNBINNED)

    def reset(self):
        self._update_state()
        return self._world_state

    def connect(self):
        return self

    def get_camera_count(self):
        return len(self.cameras)

    def get_camera_image(self, id):
        cam = self.cameras[id]
        capture = cam.get_capture(-1)
        img = cv2.resize(capture.color.data, tuple(self.obs_res))
        return img, None # TODO include depth 

    def get_camera_images(self):
        return list(self.get_camera_image(id) for id in self.cameras.keys())

    def get_world_state(self):
        return self._world_state

    def _update_state(self):
        if self.neutral_position:
            self.robot.move(self.neutral_position, max_speed=3, max_acc=1)
        self.robot.move(self.out_position, max_speed=3, max_acc=1)
        self.__stop_cameras()
        self.detector.start()
        self._world_state = self.detector.detect_keypoints(use_arm_coord=True)
        self.detector.stop()
        self.__start_cameras()
        if self.neutral_position:
            self.robot.move(self.neutral_position, max_speed=3, max_acc=1)

    def __start_cameras(self):
        for cam in self.cameras.values():
            cam.start_cameras(self.k4a_config)

    def __stop_cameras(self):
        for cam in self.cameras.values():
            cam.stop_cameras()