from mmky import writers
import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Tool to inspect MechanicalMonkey recordings.")
parser.add_argument(
    "-f",
    "--file",
    type=str,
    help="The file to play back.")

args = parser.parse_args()
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

for image in episode['images']:
    cv2.imshow("image", image)
    cv2.waitKey(16)
