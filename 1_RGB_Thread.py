#!/usr/bin/env python3

import threading
import queue
import time
from jetson_inference import poseNet
import jetson_utils

shared_keypoints = queue.Queue()
q = queue.Queue()

# RGB Thread functions

def rgb_camera_thread():

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

        # Iterate over detected poses and extract nose keypoint
        for pose in poses:

            noseidx = pose.FindKeypoint('nose')

            if noseidx < 0 :
                print("No nose")
                continue
            else:
                nose = pose.Keypoints[noseidx]
                print(nose)
            try:
                shared_keypoints.put({
                    "nose": nose
                }, block=False)
            except queue.Full:
                pass  # Queue is full, skip updating

        # Display the RGB frame with overlays
        disp.Render(rgb_frame)

        # Update the output window
        disp.SetStatus("Pose Estimation | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        # Check for exit condition
        if not disp.IsStreaming():
            shared_keypoints.put(None)  # Add sentinel value to indicate end of stream
            break


# Start the RGB thread

rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)
rgb_thread.start()
rgb_thread.join()
