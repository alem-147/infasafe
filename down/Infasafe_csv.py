#!/usr/bin/env python3

import threading
import queue
import time
import cv2
import numpy as np
from jetson_inference import poseNet
import jetson_utils
from uvctypes import *
import datetime
import csv

# Create a CSV file for data export
csv_filename = "breath_data.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "MaxVal"])


shared_regions = queue.Queue()
q = queue.Queue()

# RGB Thread functions

def rgb_camera_thread():

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

            # Add the thermal ROI to the shared_regions queue
            #if 160 < left_bound < 1120 and 160 < right_bound < 1120:
                #left_bound = left_bound
                #right_bound = right_bound
            shared_regions.put(breath_roi)

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
                    #minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
                    #img = raw_to_8bit(data)
                    #display_temperature(img, minVal, minLoc, (255, 0, 0))
                    #display_temperature(img, maxVal, maxLoc, (0, 0, 255))
                    
                    # Get the shared thermal ROI from the RGB thread
                    if not shared_regions.empty():
                        
                        thermal_roi = shared_regions.get()
                        x1, y1, x2, y2 = thermal_roi
                        x1 = int(x1-180)
                        x2 = int(x2-190)
                        y1 = int(y1-60)
                        y2 = int(y2-40)
                        #print(x1,x2,y1,y2)
                        #print(data.shape)
                        breath_rom = data[y1:y2, x2:x1]
                        #print(breath_rom)
                        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(breath_rom)
                        img = raw_to_8bit(data)
                        display_temperature(img, maxVal, (x1,y1), (0, 255, 0))
                        #display_temperature(img, average_temperature , (x1,y1), (0, 255, 0))
                        
                        # Draw the transformed ROI on the IR image
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
                        cv2.imshow('Lepton Radiometry', img)
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                        # Write timestamp and maxVal to the CSV file
                        csv_writer.writerow([timestamp, maxVal])

                    
                    #cv2.imshow('Lepton Radiometry', img)
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

rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)
ir_thread = threading.Thread(target=ir_camera_thread, daemon=True)

rgb_thread.start()
ir_thread.start()

rgb_thread.join()
ir_thread.join()


# Close the CSV file
csv_file.close()

