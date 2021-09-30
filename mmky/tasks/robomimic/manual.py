from argparse import ArgumentParser
import math
import os
import time
import numpy as np
import h5py

from mmky.tasks.robomimic.simpleenv import SimpleEnv
try:
    import inputs
except:
    print('This sample needs the inputs package (pip install inputs).')
    exit()

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

def normalize_thumb_value(v):
    # eliminate deadzone and normalize
    return (v - 8000 * v/abs(v) if abs(v) > 8000 else 0) / (32768.0 - 8000)

def get_gamepad_state():
    return inputs.devices.gamepads[0]._GamePad__read_device().gamepad

if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument('--config-path', required=True)
    # parser.add_argument('--output-dir', required=True)
    # maargs = parser.parse_args()

    env = SimpleEnv()
    exit = False
    print('Use thumbsticks and dpad to move the arm.\n'
          'Use triggers to open/close gripper.\n'
          'Press A to toggle the success flag.\n'
          'Press B to end the episode.\n'
          'Press the back button to exit.')

    hand = 255
    episode = 1    
    while not exit:
        # reset the episode
        print(f'Starting episode {episode}')
        proprios, images, actions, rewards, dones, successes = [], [], [], [], [], []
        obs = env.reset()
        proprios.append(obs['proprio'])
        images.append(obs['image'])

        # loop until the end of the episode (B button)
        success = False
        done = False
        while not done and not exit: 
            gps = get_gamepad_state()
            # the two analog thumb sticks control the first four joints
            base = -normalize_thumb_value(gps.r_thumb_x)
            shoulder = normalize_thumb_value(gps.r_thumb_y)
            elbow = -normalize_thumb_value(gps.l_thumb_y)
            wrist1 = normalize_thumb_value(gps.l_thumb_x)
            # d-pad controls wrists
            wrist2 = 1 if gps.buttons == DPAD_LEFT else -1 if gps.buttons == DPAD_RIGHT else 0
            wrist3 = 1 if gps.buttons == DPAD_UP else -1 if gps.buttons == DPAD_DOWN else 0
            
            if gps.left_trigger > hand:
                hand = gps.left_trigger
            elif gps.right_trigger > 255 - hand:
                hand = 255 - gps.right_trigger
            
            # A toggles the success flag
            if gps.buttons == BTN_A:
                while gps.buttons == BTN_A:
                    gps = get_gamepad_state()
                success = not success
                print(f"success = {success}")

            # B means the episode is done
            done = gps.buttons == BTN_B

            # Back btn exits
            exit = gps.buttons == BTN_BACK 

            action = [base, shoulder, elbow, wrist1, wrist2, wrist3, hand/255.]
            actions.append(action)
            obs, reward, _, _ = env.step(action)
            env.render()
            proprios.append(obs['proprio'])
            images.append(obs['image'])
            rewards.append(reward)
            dones.append(done)
            successes.append(success)

        episode_data = {
            'proprios': proprios,
            'images': images,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'successes': successes
        }

        #path = os.path.join(os.path.dirname(__file__), f'{episode}-{time.time()}.hdf5')
        path = os.path.join("C:\\recordings\\simpletraj", f'{episode}-{time.time()}.hdf5')
        with h5py.File(path, 'w') as f:
            for k, v in episode_data.items():
                f.create_dataset(k, data=np.stack(v))
        
        print(f"Episode saved at {path}")
        episode += 1

    env.close()