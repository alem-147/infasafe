#!/usr/bin/env python3

import threading
import queue
import time
import cv2
import numpy as np
from jetson_inference import poseNet
import jetson_utils
from uvctypes import *

shared_regions = queue.Queue()
q = queue.Queue()

# RGB Thread functions

def rgb_thread():
    # Load the pose estimation model
    pose_net = poseNet('resnet18-body', threshold=0.15)

    # Initialize the camera or video input source
    input = jetson_utils.videoSource("csi://0")
    output = jetson_utils.videoOutput("display://0")

    while True:

        # Capture the next image from the RGB camera
        rgb_frame = input.Capture()

        if rgb_frame is None:
            continue

        # Perform pose estimation on the RGB frame
        poses = pose_net.Process(rgb_frame)

        # Clear previous thermal ROIs
        shared_regions.queue.clear()

        # Iterate over detected poses and extract nose keypoint
        for pose in poses:

            nose_idx = pose.FindKeypoint('nose')
            leye_idx = pose.FindKeypoint('left_eye')
            reye_idx = pose.FindKeypoint('right_eye')

            thermal_roi = (0,0,0,0)
            breath_roi = (0,0,0,0)

            if nose_idx < 0:
                continue
            if leye_idx > 0 and reye_idx > 0:
                reye = pose.Keypoints[reye_idx]
                leye = pose.Keypoints[leye_idx]
                nose = pose.Keypoints[nose_idx]
                # dealing with reversable values -> posenet doesn't deal with upsideown
                # left_bound, right_bound = (leye.x, reye.x) if leye.x < reye.x else (reye.x, leye.x)

                # can always assume posenet will have left eye to the left of right
                # this is presumeably due to training data
                left_bound, right_bound = leye.x, reye.x
                upper_bound, lower_bound = (min(leye.y,reye.y),nose.y)

                # breathROI=((left_bound, nose.y, right_bound,nose.y+50))
            elif reye_idx > 0:
                nose = pose.Keypoints[nose_idx]
                reye = pose.Keypoints[reye_idx]
                # left_bound, right_bound = (nose.x, reye.x) if nose.x < reye.x else (reye.x, nose.x)
                # upper_bound, lower_bound = (min(reye.y, nose.y),max(reye.y,nose.y))

                left_bound, right_bound = nose.x, reye.x
                upper_bound, lower_bound = reye.y, nose.y
            elif leye_idx > 0:
                nose = pose.Keypoints[nose_idx]
                leye = pose.Keypoints[leye_idx]
                # left_bound, right_bound = (nose.x, leye.x) if nose.x < leye.x else (leye.x, nose.x)
                # upper_bound, lower_bound = (min(leye.y, nose.y),max(leye.y,nose.y))

                left_bound, right_bound = leye.x, nose.x
                upper_bound, lower_bound = leye.y, nose.y
            # neither is seen
            else:
                left_bound, right_bound = (nose.x-50, nose.x+50)
                upper_bound, lower_bound = (nose.y-50,nose.y+50)

               
            thermal_roi = (left_bound, upper_bound, right_bound, lower_bound)
            breath_roi = (left_bound,nose.y,right_bound,nose.y+(nose.y-upper_bound)*.75)
            print(thermal_roi)

            # Add the thermal ROI to the shared_regions queue
            shared_regions.put(thermal_roi)

            # Draw rectangles on the RGB frame to highlight thermal ROIs
            # for thermal_roi in shared_regions.queue:
            jetson_utils.cudaDrawRect(rgb_frame, thermal_roi, (255, 127, 0, 200))
            jetson_utils.cudaDrawRect(rgb_frame, breath_roi, (0, 127, 255, 200))

        # Display the RGB frame with overlays
        output.Render(rgb_frame)

        # Update the output window
        output.SetStatus("Pose Estimation | Network {:.0f} FPS".format(pose_net.GetNetworkFPS()))

        # Check for exit condition
        if not output.IsStreaming():
            shared_regions.put(None)  # Add sentinel value to indicate end of stream
            break

# IR Thread functions

def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(
        array_pointer.contents, dtype=np.dtype(np.uint16)
    ).reshape(
        frame.contents.height, frame.contents.width
    )

    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return

    if not q.full():
        q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
    return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
    return (val - 27315) / 100.0

def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img, "{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

# Function to transform coordinates from RGB to IR image space placeholder
def transform_coordinates(region):
    # Constants
    RGB_WIDTH, RGB_HEIGHT = 1280, 720
    IR_WIDTH, IR_HEIGHT = 640, 480

    # Unpack region
    x1, y1, x2, y2 = region

    # Scale the coordinates
    x1_ir = int(x1 * IR_WIDTH / RGB_WIDTH)
    y1_ir = int(y1 * IR_HEIGHT / RGB_HEIGHT)
    x2_ir = int(x2 * IR_WIDTH / RGB_WIDTH)
    y2_ir = int(y2 * IR_HEIGHT / RGB_HEIGHT)

    return (x1_ir, y1_ir, x2_ir, y2_ir)

#IR Thread

def ir_camera_thread():
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(1)

    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print("uvc_find_device error")
            exit(1)

        try:
            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0:
                print("uvc_open error")
                exit(1)

            print("Device opened!")

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                print("Device does not support Y16")
                exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                                                  frame_formats[0].wWidth, frame_formats[0].wHeight,
                                                  int(1e7 / frame_formats[0].dwDefaultFrameInterval))

            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
                print("uvc_start_streaming failed: {0}".format(res))
                exit(1)

            try:
                while True:
                    data = q.get(True, 500)
                    if data is None:
                        break
                    data = cv2.resize(data[:, :], (640, 480))
                    minVal, maxVal, _, _ = cv2.minMaxLoc(data)
                    average_temperature = ktoc(np.mean(data))
                    print("Average Temperature: {:.2f} °C".format(average_temperature))
                    img = raw_to_8bit(data)
                    display_temperature(img, minVal, (0, 0), (255, 0, 0))
                    display_temperature(img, maxVal, (img.shape[1] - 200, 0), (0, 0, 255))
                    cv2.imshow('Lepton Radiometry', img)
                    cv2.waitKey(1)

                cv2.destroyAllWindows()
            finally:
                libuvc.uvc_stop_streaming(devh)

            print("Done")
        finally:
            libuvc.uvc_unref_device(dev)
    finally:
        libuvc.uvc_exit(ctx)

# Start the RGB/IR threads

rgb_thread = threading.Thread(target=rgb_thread, daemon=True)
ir_thread = threading.Thread(target=ir_camera_thread, daemon=True)

rgb_thread.start()
ir_thread.start()

rgb_thread.join()
ir_thread.join()


