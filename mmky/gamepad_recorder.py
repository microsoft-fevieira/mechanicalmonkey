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
from roman import *
import mmky.k4a as k4a

DPAD_UP = 1
DPAD_DOWN = 2
DPAD_LEFT = 4
DPAD_RIGHT = 8
BTN_MENU = 16
BTN_BACK = 32
THUMB_STICK_PRESS_LEFT = 64
THUMB_STICK_PRESS_RIGHT = 128
BTN_SHOULDER_LEFT = 256
BTN_SHOULDER_RIGHT = 512
BTN_A = 4096
BTN_B = 8192
BTN_X = 16384
BTN_Y = 32768

# keyboard to gamepad mapping
keymap = {
    DPAD_UP: 'E',
    DPAD_DOWN: 'D',
    DPAD_LEFT: 'S',
    DPAD_RIGHT: 'F',
    BTN_MENU: 'home',
    BTN_BACK: 'backspace',
    THUMB_STICK_PRESS_LEFT: 'shift+space',
    THUMB_STICK_PRESS_RIGHT: 'space',
    BTN_SHOULDER_LEFT: 'ctrl',
    BTN_SHOULDER_RIGHT: 'alt',
    BTN_A: 'A',
    BTN_B: 'B',
    BTN_X: 'X',
    BTN_Y: 'Y'
}

class gamepad_or_keyboard():
    def __init__(self, gamepad_id=0):
        self.gamepad_id = gamepad_id
        self.gps = None

    def normalize_thumb_value(self, v):
        # eliminate deadzone and normalize
        return (v - 8000 * v/abs(v) if abs(v) > 8000 else 0) / (32768.0 - 8000)

    def refresh(self):
        if len(inputs.devices.gamepads) > self.gamepad_id:
            self.gps = inputs.devices.gamepads[self.gamepad_id]._GamePad__read_device().gamepad

    def button_pressed(self, btn):
        return self.gps.buttons & btn if self.gps else keyboard.is_pressed(keymap[btn])

    def button_toggled(self, btn, prev_state=False):
        if self.button_pressed(btn):
            while self.button_pressed(btn):
                self.refresh()
            return not prev_state
        return prev_state

    def left_trigger(self):
        return self.gps.left_trigger if self.gps else 255 * keyboard.is_pressed(',')

    def right_trigger(self):
        return self.gps.right_trigger if self.gps else 255 * keyboard.is_pressed('.')

    def r_thumb_x(self):
        return self.normalize_thumb_value(self.gps.r_thumb_x) if self.gps else 0 if keyboard.is_pressed('shift') else 1 if keyboard.is_pressed('right') else -1 if keyboard.is_pressed('left') else 0

    def r_thumb_y(self):
        # since right thumb up/down is reversed (like airplane pitch), keyboard mapping is negated
        return self.normalize_thumb_value(self.gps.r_thumb_y) if self.gps else 0 if keyboard.is_pressed('shift') else -1 if keyboard.is_pressed('up') else 1 if keyboard.is_pressed('down') else 0

    def l_thumb_x(self):
        return self.normalize_thumb_value(self.gps.l_thumb_x) if self.gps else 1 if keyboard.is_pressed('shift+right') else -1 if keyboard.is_pressed('shift+left') else 0

    def l_thumb_y(self):
        return self.normalize_thumb_value(self.gps.l_thumb_y) if self.gps else 1 if keyboard.is_pressed('shift+up') else -1 if keyboard.is_pressed('shift+down') else 0

if __name__ == '__main__':
    use_sim=True
    joint_gain = 0.1
    pose_gain = 0.02
    xy_flipped = False

    robot = connect(use_sim=use_sim)
    gk = gamepad_or_keyboard()
    done = False
    print('*****************************************.')
    print(
        'Thumbsticks: primary way to move the arm (EEF position control in cylindrical coordinates).\n'
        'Thumbsticks + right shoulder button: primary way to rotate the gripper (joint control).\n'
        'D-Pad: alternate way to move the arm (x-y position control in cartesian coordinates).\n'
        'Thumbsticks + left shoulder button: alternate way to rotate gripper (EEF position control, roll/pitch/yaw).\n'
        'Triggers: open/close gripper.\n'
        'A: start recording.\n'
        'B: stop recording.\n'
        'X: swap x/y axes on D-Pad.\n'
        'Y: change the grasp mode (pinch vs basic).\n'
        'Menu button: move to home position.\n'
        'Esc: exit.'
    )

    assert(not robot.is_moving())
    def move(target, duration=0.01, max_speed=2, max_acc=1):
        # perform the action
        if use_sim:
            # hack, move_rt doesn't yet work well in sim
            robot.move(target, max_speed=1, max_acc=1, timeout=0.0)
        else:
            robot.move_rt(target, duration=0.01, max_speed=1, max_acc=1, timeout=0.0)
    home = Joints(0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0)
    tangent_move_reference = (0, 0)
    t0 = time.time()
    while not done:
        # print(time.time()-t0)
        t0 = time.time()
        gk.refresh()
        xy_flipped = gk.button_toggled(BTN_X, xy_flipped)

        if gk.button_pressed(BTN_MENU):
            move(home)
        elif gk.button_pressed(BTN_SHOULDER_RIGHT):
            # wrist control in joint positions. Direction is optimized for the gripper-down position
            wrist1 = joint_gain * gk.r_thumb_y()
            wrist2 = -joint_gain * gk.l_thumb_x()
            wrist3 = joint_gain * gk.r_thumb_x()
            target = robot.arm.state.joint_positions() + [0, 0, 0, wrist1, wrist2, wrist3]
            move(target)
        elif gk.button_pressed(BTN_SHOULDER_LEFT):
            # wrist control, roll/pitch/yaw
            roll = joint_gain * gk.l_thumb_x()
            pitch = joint_gain * -gk.r_thumb_y()
            yaw = joint_gain * -gk.r_thumb_x()
            target = robot.arm.state.tool_pose().to_xyzrpy() + [0, 0, 0, roll, pitch, yaw]
            target = Tool.from_xyzrpy(target)
            move(target)
        elif gk.button_pressed(DPAD_DOWN) or gk.button_pressed(DPAD_LEFT) or gk.button_pressed(DPAD_RIGHT) or gk.button_pressed(DPAD_UP):
            # x-y move in robot base coordinate system
            dx = pose_gain * (-1 if gk.button_pressed(DPAD_LEFT) else 1 if gk.button_pressed(DPAD_RIGHT) else 0)
            dy = pose_gain * (1 if gk.button_pressed(DPAD_UP) else -1 if gk.button_pressed(DPAD_DOWN) else 0)
            if xy_flipped:
                tmp = dx
                dx = dy
                dy = -tmp

            dz = pose_gain * -gk.r_thumb_y()
            target = robot.arm.state.tool_pose() + [dx, dy, dz, 0, 0, 0]
            move(target)
        else:
            da = -gk.r_thumb_x()
            dz = pose_gain * -gk.r_thumb_y()
            dd = pose_gain * gk.l_thumb_y()
            dt = -pose_gain * gk.l_thumb_x()
            (x, y, z, roll, pitch, yaw) = robot.arm.state.tool_pose().to_xyzrpy()
            a = math.atan2(y, x)
            d = math.sqrt(x*x + y*y)

            if dt == 0:
                # move in cylindrical coordinates
                a = a + da
                d = d + dd
                yaw = yaw + da
                tangent_move_reference = (a, d)
            else:
                # tangent move
                ta, td = tangent_move_reference
                tt = math.sin(a - ta) + dt
                td = td + dd
                da = math.atan2(tt, td)
                a = ta + da
                d = math.sqrt(tt*tt + td*td)
                tangent_move_reference = (ta, td)

            x = d * math.cos(a)
            y = d * math.sin(a)
            z = z + dz
            target = Tool.from_xyzrpy([x, y, z, roll, pitch, yaw])
            move(target)
            #robot.move(target, max_speed=2, max_acc=1, timeout=0.0)

        if gk.left_trigger() > 0 or gk.right_trigger() > 0:
            if gk.left_trigger() > 30:
                robot.open(speed=gk.left_trigger(), timeout=0)
            elif gk.right_trigger() > 30:
                robot.grasp(speed=gk.right_trigger(), timeout=0)
            else:
                robot.hand.stop()

        # Y btn changes grasp
        if gk.button_toggled(BTN_Y):
            mode = GraspMode.PINCH if robot.hand.state.mode() != GraspMode.PINCH else GraspMode.BASIC
            robot.hand.set_mode(mode)

        # Back btn exits
        done = keyboard.is_pressed('esc')
        time.sleep(0.01)
    robot.disconnect()
