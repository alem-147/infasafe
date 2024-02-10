#!/usr/bin/env python3

import multiprocessing
from jetson_inference import poseNet
import jetson_utils

# RGB Process functions

def rgb_camera_process():

    # Load the pose estimation model
    net = poseNet('resnet18-body', threshold=0.15)

    # Initialize the camera or video input source
    cam = jetson_utils.videoSource("csi://0")
    disp = jetson_utils.videoOutput("display://0")

    while True:

        # Capture the next image from the RGB camera
        rgb_frame = cam.Capture()

        if rgb_frame is None:
            continue

        # Perform pose estimation on the RGB frame
        poses = net.Process(rgb_frame)

        print("detected {:d} objects in image".format(len(poses)))

        # Iterate over detected poses and extract nose keypoint
        for pose in poses:
            print(pose)
            print(pose.Keypoints)
            print('Links', pose.Links)

        # Display the RGB frame with overlays
        disp.Render(rgb_frame)

        # Update the output window
        disp.SetStatus("Pose Estimation | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        net.PrintProfilerTimes()

        # Check for exit condition
        if not cam.IsStreaming() or not disp.IsStreaming():
            break

if __name__ == '__main__':
    rgb_process = multiprocessing.Process(target=rgb_camera_process)
    rgb_process.start()

