from mmky import writers
import os
import cv2
import argparse
import numpy as np

try:
    import imageio as iio
except:
    print("To use this, you need to pip install imageio, imageio-ffmpeg")


parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Tool to convert MechanicalMonkey recordings to videos.")
parser.add_argument(
    "-f",
    "--file",
    type=str,
    help="The file to play back.")
parser.add_argument(
    "-v",
    "--video",
    type=str,
    help="The video file to create.")
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="The framerate of the generated video.")
# parser.add_argument(
#     "-c",
#     "--cam",
#     type=str,
#     help="Camera to convert to video.")
args = parser.parse_args()
ext = os.path.splitext(args.file)[-1]
if ext == '.hdf5':
    episode = writers.readRobosuite(args.file)
elif ext == '.npy':
    episode = writers.readSimpleNpy(args.file)
else:
    raise Exception(f"Unsuported file type {ext}. Supported files are hdf5 and npy")

writer = iio.get_writer(args.video, fps=30)
for image in episode['images']:
    cv2.imshow("image", image)
    writer.append_data(image)
    cv2.waitKey(2)
writer.close()