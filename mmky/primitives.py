import math
import numpy as np
import random
from roman import Robot, Tool, Joints, GraspMode


def go_to_start(robot: Robot, start_pose, grasp_mode=GraspMode.PINCH, max_speed=1, max_acc=1):
    if not robot.move(start_pose, max_speed=max_speed, max_acc=max_acc, timeout=10):
        return False
    if not robot.release(timeout=2):
        return False
    robot.set_hand_mode(grasp_mode)
    return True

def pick(robot, grasp_height, max_speed=1, max_acc=1, max_time=10):
    back = robot.tool_pose
    pick_pose = back.clone()
    pick_pose[Tool.Z] = grasp_height
    if not robot.open(timeout=max_time):
        return False
    if not robot.move(pick_pose, max_speed=max_speed, max_acc=max_acc, timeout=max_time):
        return False
    if not robot.grasp(timeout=max_time):
        return False
    if not robot.move(back, max_speed=max_speed, max_acc=max_acc, timeout=max_time):
        return False
    return robot.has_object

def place(robot, release_height, max_speed=0.5, max_acc=0.5, max_time=10):
    back = robot.tool_pose
    release_pose = back.clone()
    release_pose[Tool.Z] = release_height
    if not robot.touch(release_pose, timeout=max_time, max_speed=0.5, max_acc=0.5):
        return False
    if not robot.release(128, timeout=max_time):
        return False
    if not robot.move(back, max_speed=0.5, max_acc=0.5, timeout=max_time):
        return False
    return not robot.has_object

def pivot_xy(robot, reference_pose: Tool, x, y, dr, max_speed=0.3, max_acc=1, max_time=10):
    target = np.array(reference_pose.to_xyzrpy())
    target[:2] = x, y
    target[5] = math.atan2(y, x) + math.pi / 2 # yaw, compensating for robot config offset (base offset is pi, wrist offset from base is -pi/2)
    jtarget = robot.get_inverse_kinematics(Tool.from_xyzrpy(target))
    jtarget[Joints.WRIST3] = robot.joint_positions[Joints.WRIST3] + 0.3 * dr
    return robot.move(jtarget, max_speed=max_speed, max_acc=max_acc, timeout=max_time)

def move_dxdy(robot, reference_z, dx, dy, dr, max_speed=0.1, max_acc=1):
    pose = robot.tool_pose
    pose = pose + [0.01 * dx, 0.01 * dy, 0, 0, 0, 0]
    pose[Tool.Z] = reference_z
    jtarget = robot.get_inverse_kinematics(pose)
    jtarget[Joints.WRIST3] = robot.joint_positions[Joints.WRIST3] + 0.3 * dr
    robot.move(jtarget, max_speed=max_speed, max_acc=max_acc, timeout=0)
    return False

def pivot_dxdy(robot, reference_pose: Tool, dx, dy, dr, max_speed=0.3, max_acc=1):
    pose = robot.tool_pose
    joints = robot.joint_positions

    if dx or dy:
        x = pose[Tool.X] + 0.01 * dx
        y = pose[Tool.Y] + 0.01 * dy
        target = np.array(reference_pose.to_xyzrpy())
        target[2:] = x, y
        target[5] = math.atan2(y, x) + math.pi / 2 # yaw, compensating for robot config offset (base offset is pi, wrist offset from base is -pi/2)
        jtarget = robot.get_inverse_kinematics(Tool.from_xyzrpy(target))
    else:
        jtarget = joints.clone()
    jtarget[Joints.WRIST3] = joints[Joints.WRIST3] + 0.3 * dr
    robot.move(jtarget, max_speed=max_speed, max_acc=max_acc, timeout=0)
    return True


def generate_random_xy(min_angle_in_rad, max_angle_in_rad, min_dist, max_dist):
    # Sample a random distance from the coordinate origin (i.e., arm base) and a random angle.
    dist = min_dist + random.random() * (max_dist - min_dist)
    angle = min_angle_in_rad + random.random() * (max_angle_in_rad - min_angle_in_rad)
    return [dist * math.cos(angle), dist * math.sin(angle)]

