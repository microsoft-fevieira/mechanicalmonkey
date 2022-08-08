try:
    import inputs
except:
    print('This script needs the inputs package (pip install inputs).')
    exit()
try:
    import keyboard
except:
    print('This script needs the keyboard package (pip install keyboard).')
    exit()

import time
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import random

from numpy import set_printoptions
set_printoptions(precision=2, threshold=5, edgeitems=4, suppress=True)

from roman import *
from pyquaternion import Quaternion
from oculus_reader.reader import OculusReader

import mmky.utils.transformation_utils as tr

def oculus_to_robot(current_vr_transform):
    current_vr_transform = tr.RpToTrans(Quaternion(axis=[1, 0, 0], angle=math.pi / 2).rotation_matrix, np.zeros(3)).dot(current_vr_transform)

    return current_vr_transform

def get_pose_and_button(reader):
    poses, buttons = reader.get_transformations_and_buttons()

    if 'r' not in poses:
        return None, None, None, None, None, None

    return poses['r'], buttons['RTr'], buttons['rightTrig'][0], buttons['RG'], buttons['A'], buttons['B']

def move(robot, target, use_sim=True, duration=0.01, max_speed=1, max_acc=1):
    # perform the action
    if use_sim:
        # hack, move_rt doesn't yet work well in sim
        robot.move(target, max_speed=max_speed, max_acc=max_acc, timeout=0.0)
    else:
        robot.move_rt(target, duration=duration, max_speed=max_speed, max_acc=max_acc, timeout=0.0)

def get_pose_from_h_matrix(H):
    R, p = tr.TransToRp(current_vr_transform)
    R = Rotation.from_matrix(R).as_euler('xyz').tolist()
    return np.array(p.tolist() + R)

def oculus_to_robot(pose):
    x, y, z, roll, pitch, yaw = pose
    return np.array([-x, z, y, yaw, -roll, -pitch])

if __name__ == '__main__':
    use_sim=True
    JOINT_GAIN = 1
    POSE_GAIN = 0.625

    robot = connect(use_sim=use_sim)
    reader = OculusReader()
    done = False

    assert(not robot.is_moving())
    home = Joints(0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0)

    t0 = time.time()
    reference_vr_pose = None
    reference_robot_pose = None
    prev_handle_press = False

    while not done:
        # print(time.time()-t0)
        t0 = time.time()
        current_vr_transform, trigger, trigger_continuous, handle_button, A_button, B_button = get_pose_and_button(reader)

        if B_button:
            move(robot, home)
            continue

        if current_vr_transform is None:
            continue
        else:
            if not prev_handle_press and handle_button:
                reference_vr_pose = get_pose_from_h_matrix(current_vr_transform)
                reference_robot_pose = robot.arm.state.tool_pose().to_xyzrpy()
                prev_handle_press = True

            if not handle_button:
                reference_vr_pose = None
                prev_handle_press = False
                reference_robot_pose = None
                continue

        prev_handle_press = True

        if trigger:
            robot.grasp(int((trigger_continuous) * 255), timeout=0)
        else:
            robot.open(0, timeout=0)

        delta_vr_pose = get_pose_from_h_matrix(current_vr_transform) - reference_vr_pose
        delta_robot_pose = oculus_to_robot(delta_vr_pose)

        dx, dy, dz = delta_robot_pose[:3] * POSE_GAIN
        droll, dpitch, dyaw = delta_robot_pose[3:] * JOINT_GAIN

        target = reference_robot_pose + [dx, dy, dz, droll, dpitch, dyaw]

        if A_button:
            print(robot.arm.state.tool_pose().to_xyzrpy())
            print(target)

        move(robot, Tool.from_xyzrpy(target), max_speed=2, max_acc=2)

        time.sleep(0.01)

    robot.disconnect()
