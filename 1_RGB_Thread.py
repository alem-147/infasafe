#!/usr/bin/env python3

import threading
import queue
import time
from jetson_inference import poseNet
import jetson_utils

# RGB Thread functions

def rgb_camera_thread():

    # Load the pose estimation model
    net = poseNet('resnet18-body', threshold=0.15)

    # Initialize the camera or video input source
    cam = jetson_utils.videoSource("/dev/video0")
    disp = jetson_utils.videoOutput("webrtc://10.0.0.209:8554/live")

    while True:

        # Capture the next image from the RGB camera
        rgb_frame = cam.Capture()

        if rgb_frame is None:
            continue

        # Perform pose estimation on the RGB frame
        poses = net.Process(rgb_frame)

        # Display the RGB frame with overlays
        disp.Render(rgb_frame)

        # Update the output window
        disp.SetStatus("Pose Estimation | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        # Check for exit condition
        if not disp.IsStreaming() or not cam.IsStreaming():
            break


# Start the RGB thread

rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)
rgb_thread.start()
rgb_thread.join()
