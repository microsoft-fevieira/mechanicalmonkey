from mmky import writers
from mmky.env import RoboSuiteEnv
import os
import time
import cv2
import argparse
import numpy as np
from mmky.tasks.drop.dropreal import DropReal
from mmky.tasks.drop.dropsim import DropSim
from mmky.tasks.place.placereal import PlaceReal
from mmky.tasks.place.placesim import PlaceSim
from mmky.tasks.pour.pourreal import PourReal
from mmky.tasks.pour.poursim import PourSim
from mmky.tasks.push.pushreal import PushReal
from mmky.tasks.push.pushsim import PushSim
from mmky.tasks.stack.stackreal import StackReal
from mmky.tasks.stack.stacksim import StackSim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Tool to replay MechanicalMonkey recordings.")
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="The task to initialize.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="The file to replay.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="The config file to load.")

    args = parser.parse_args()
    simtype = eval(args.task + "Sim")
    realtype = eval(args.task + "Real")
    env = RoboSuiteEnv(simtype, realtype, config=args.config)
    
    ext = os.path.splitext(args.file)[-1]
    if ext == '.hdf5':
        episode = writers.readRobosuite(args.file)
    elif ext == '.npy':
        episode = writers.readSimpleNpy(args.file)
    else:
        raise Exception(f"Unsuported file type {ext}. Supported files are hdf5 and npy")

    print(f"Episode length: {len(episode['images'])}")
    print(f"Available data: {[k for k in episode.keys()]}")
    print(f"Image resolution: {episode['images'][0].shape}")
    actions = np.array(episode['actions'])
    print(f"Action range: [{actions.min(axis=0)}, {actions.max(axis=0)}]")
    print(f"Reward range: [{np.min(episode['rewards'])}, {np.max(episode['rewards'])}]")
    print(f"Episode return: {np.sum(episode['rewards'])}")
    print(f"Success steps: {np.count_nonzero(episode['successes'])}")
    print(f"Done steps: {np.count_nonzero(episode['dones'])}")
    

    #env.reset(start_pose=episode['proprios'][0][18:20])
    env.reset()
    print("reset is done")
    cv2.imshow("start image", episode['images'][0])
    cv2.waitKey(16)
    input("Arrange the scene as in the image and press enter.")

    for action in episode['actions']:
        env.step(action)
        time.sleep(1/60.)

    env.close()