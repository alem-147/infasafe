#!/usr/bin/env python3

import threading
import queue
import time
import numpy as np
from jetson_inference import poseNet
import jetson_utils
from uvctypes import *
import cv2

shared_keypoints = queue.Queue()
q = queue.Queue()

# RGB Thread functions

def rgb_camera_thread():

    # Load the pose estimation model
    net = poseNet('densenet121-body', threshold=0.15)

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

        # Clear previous thermal ROIs
        shared_keypoints.queue.clear()

        # Iterate over detected poses and extract nose keypoint
        for pose in poses:

            noseidx = pose.FindKeypoint('nose')
            leyeidx = pose.FindKeypoint('left_eye')
            reyeidx = pose.FindKeypoint('right_eye')

            if noseidx < 0 :
                print("No nose")
                continue
            if leyeidx > 0 and reyeidx > 0:
                reye = pose.Keypoints[reyeidx]
                leye = pose.Keypoints[leyeidx]
                nose = pose.Keypoints[noseidx]
            elif reyeidx > 0:
                nose = pose.Keypoints[noseidx]
                reye = pose.Keypoints[reyeidx]
            elif leyeidx > 0:
                nose = pose.Keypoints[noseidx]
                leye = pose.Keypoints[leyeidx]
            else:
                nose = pose.Keypoints[noseidx]
            try:
                shared_keypoints.put({
                    "nose": nose,
                    "left_eye": leye,
                    "right_eye": reye
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

# IR Thread functions

def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.fromiter(
    frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes).reshape(frame.contents.height, frame.contents.width, 2)

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
    cv2.normalize(data, data, 0, 255, cv2.NORM_MINMAX)
    #np.right_shift(data, 8, data)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img, "{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

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
                    data = cv2.resize(data[:, :], (960, 720))
                    data = cv2.flip(data, 0)
                    data = cv2.flip(data, 1)
                    cv2.normalize(data, data, 0, 255, cv2.NORM_MINMAX)
                    data = data.astype(np.uint8)
                    img = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
                    
                    # Get the shared thermal ROI from the RGB thread
                    try:
                        shared_keypoint_data = shared_keypoints.get(block=False)
                        nose = shared_keypoint_data["nose"]
                        left_eye = shared_keypoint_data["left_eye"]
                        right_eye = shared_keypoint_data["right_eye"]

                        # Use the keypoints as needed
                        if nose is not None:
                            # Perform actions using nose keypoint
                            cv2.circle(img, nose, 1, (0, 255, 0), -1)
                            # cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
                            pass
                        if left_eye is not None:
                            # Perform actions using left eye keypoint
                            cv2.circle(img, left_eye, 1, (255, 0, 0), -1)
                            pass
                        if right_eye is not None:
                            # Perform actions using right eye keypoint
                            cv2.circle(img, right_eye, 1, (0, 0, 255), -1)
                            pass

                    except queue.Empty:
                        pass  # Queue is empty, continue without using keypoints
                    
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

#rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)
ir_thread = threading.Thread(target=ir_camera_thread, daemon=True)

#rgb_thread.start()
ir_thread.start()

#rgb_thread.join()
ir_thread.join()

