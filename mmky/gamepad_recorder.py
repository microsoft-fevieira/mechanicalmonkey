import time
try:
    import inputs
except:
    print('This sample needs the inputs package (pip install inputs).')
    exit()
try:
    import keyboard
except:
    print('This sample needs the keyboard package (pip install keyboard).')
    exit()

from roman import *
import math
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

    def refresh(self, id=0):
        if len(inputs.devices.gamepads) > self.gamepad_id:
            self.gps = inputs.devices.gamepads[id]._GamePad__read_device().gamepad

    def button_pressed(self, btn):
        return self.gps.buttons & btn if self.gps else keyboard.is_pressed(keymap[btn])

    def left_trigger(self):
        return self.gps.left_trigger if self.gps else 255 * keyboard.is_pressed('shift+enter')

    def right_trigger(self):
        return self.gps.right_trigger if self.gps else 255 * keyboard.is_pressed('enter')

    def r_thumb_x(self):
        return self.normalize_thumb_value(self.gps.r_thumb_x) if self.gps else 0 if keyboard.is_pressed('shift') else 1 if keyboard.is_pressed('right') else -1 if keyboard.is_pressed('left') else 0

    def r_thumb_y(self):
        return self.normalize_thumb_value(self.gps.r_thumb_y) if self.gps else 0 if keyboard.is_pressed('shift') else 1 if keyboard.is_pressed('up') else -1 if keyboard.is_pressed('down') else 0

    def l_thumb_x(self):
        return self.normalize_thumb_value(self.gps.l_thumb_x) if self.gps else 1 if keyboard.is_pressed('shift+right') else -1 if keyboard.is_pressed('shift+left') else 0

    def l_thumb_y(self):
        return self.normalize_thumb_value(self.gps.l_thumb_y) if self.gps else 1 if keyboard.is_pressed('shift+up') else -1 if keyboard.is_pressed('shift+down') else 0

if __name__ == '__main__':
    use_sim=True
    robot = connect(use_sim=use_sim, config={"hand.activate": False})
    gk = gamepad_or_keyboard()
    done = False
    print('*****************************************.')
    print(
        'Thumbsticks: primary way to move the arm (EEF position control in cylindrical coordinates).'
        'Thumbsticks + right shoulder button: primary way to rotate the gripper (joint control).'
        'D-Pad: alternate way to move the arm (x-y position control in cartesian coordinates).'
        'Thumbsticks + left shoulder button: alternate way to rotate gripper (EEF position control, roll/pitch/yaw).'
        'Triggers: open/close gripper.'
        'A / B: start/stop recording'
        'X: move to home position.'
        'Y: change the grasp mode (pinch vs basic).'
        'Menu button: enable/disable free drive (kinestetic control).'
        'Esc: exit.'
    )

    assert(not robot.is_moving())
    def move(target, duration=0.01, max_speed=2, max_acc=1):
        if use_sim:
            # hack, move_rt doesn't yet work well in sim
            robot.move(target, max_speed=2, max_acc=1, timeout=0.0)
        else:
            robot.move_rt(target, duration=0.01, max_speed=2, max_acc=1, timeout=0.0)

    t0 = time.time()
    while not done:
        # print(time.time()-t0)
        t0 = time.time()
        gk.refresh()

        if gk.button_pressed(BTN_SHOULDER_RIGHT):
            # wrist control, joint speeds
            wrist1 = -gk.l_thumb_y()
            wrist2 = gk.l_thumb_x()
            wrist3 = -gk.r_thumb_x()
            target = JointSpeeds(base=0, shoulder=0, elbow=0, wrist1=wrist1, wrist2=wrist2, wrist3=wrist3)
            move(target, max_acc=1)
        elif gk.button_pressed(BTN_SHOULDER_LEFT):
            # wrist control, roll/pitch/yaw
            roll = 0.1 * gk.l_thumb_x()
            pitch = 0.1 * -gk.r_thumb_y()
            yaw = 0.1 * -gk.r_thumb_x()
            target = robot.arm.state.tool_pose().to_xyzrpy() + [0, 0, 0, roll, pitch, yaw]
            target = Tool.from_xyzrpy(target)
            move(target, duration=0.01, max_speed=2, max_acc=1)
        elif gk.button_pressed(DPAD_DOWN) or gk.button_pressed(DPAD_LEFT) or gk.button_pressed(DPAD_RIGHT) or gk.button_pressed(DPAD_UP):
            # x-y move in robot base coordinate system
            dx = 0.02 * (-1 if gk.button_pressed(DPAD_LEFT) else 1 if gk.button_pressed(DPAD_RIGHT) else 0)
            dy = 0.02 * (1 if gk.button_pressed(DPAD_UP) else 1 if gk.button_pressed(DPAD_DOWN) else 0)
            dz = 0.02 * -gk.r_thumb_y()
            target = robot.arm.state.tool_pose() + [dx, dy, dz, 0, 0, 0]
            move(target, duration=0.01, max_speed=2, max_acc=1)
        else:
            # move in cylindrical coordinates
            da = -gk.r_thumb_x()
            dz = 0.02 * -gk.r_thumb_y()
            dd = 0.02 * gk.l_thumb_y()
            dt = 0.02 * gk.l_thumb_x()
            b = robot.arm.state.joint_positions()[Joints.BASE]
            b = b + da
            (x, y, z, roll, pitch, yaw) = robot.arm.state.tool_pose().to_xyzrpy()
            yaw = yaw + da
            a = math.atan2(y, x)
            d = math.sqrt(x*x + y*y)
            a = a + da
            d = d + dd
            x = d * math.cos(a)
            y = d * math.sin(a)
            z = z + dz
            target = Tool.from_xyzrpy([x, y, z, roll, pitch, yaw])
            move(target, duration=0.01, max_speed=2, max_acc=1)
            #robot.move(target, max_speed=2, max_acc=1, timeout=0.0)

        if gk.left_trigger() > robot.hand.state.position():
            robot.open(gk.left_trigger(), timeout=0)
        elif gk.right_trigger() > 255 - robot.hand.state.position():
            robot.grasp(255 - gk.right_trigger(), timeout=0)

        # Y btn changes grasp
        if gk.button_pressed(BTN_Y):
            mode = GraspMode.PINCH if robot.hand.state.mode() != GraspMode.PINCH else GraspMode.BASIC
            robot.hand.set_mode(mode)

        # Back btn exits
        done = keyboard.is_pressed('esc')
        time.sleep(0.01)
    robot.disconnect()
