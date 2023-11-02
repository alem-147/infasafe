#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import queue
import time
import numpy as np
from jetson_inference import poseNet
import jetson_utils
from uvctypes import *
import cv2
import platform
import board
import adafruit_ahtx0
import datetime
import csv

# Create a CSV file for data export
csv_filename = "breath_data.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "AvgVal"])

# Create sensor object, communicating over the board's default I2C bus
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_ahtx0.AHTx0(i2c)

# Initialize variables to store temperature and humidity values
current_temperature = 0.0
current_humidity = 0.0

BUF_SIZE = 2
q = queue.Queue(BUF_SIZE)
shared_keypoints = queue.Queue()

# Create a function to update temperature and humidity values
def update_sensor_values():
    global current_temperature, current_humidity
    while True:
        current_temperature = sensor.temperature
        current_humidity = sensor.relative_humidity
        time.sleep(2)

# RGB Thread functions

def rgb_camera_thread():

    # Load the pose estimation model
    net = poseNet('densenet121-body', threshold=0.15)

    # Initialize the camera or video input source
    cam = jetson_utils.videoSource("csi://0")
    disp = jetson_utils.videoOutput("display://0")

    # Create a font for text overlay
    font = jetson_utils.cudaFont()

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
                    "leye": leye,
                    "reye": reye
                }, block=False)
            except queue.Full:
                pass  # Queue is full, skip updating

        # Overlay temperature and humidity values on the frame
        font.OverlayText(rgb_frame, text=f"Temp: {current_temperature:.1f}°C   Humidity: {current_humidity:.1f}%", x=5, y= 5 + (font.GetSize() + 5), color=font.White, background=font.Gray40)

        # Display the RGB frame with overlays
        disp.Render(rgb_frame)

        # Update the output window
        disp.SetStatus("Pose Estimation | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        # Check for exit condition
        if not disp.IsStreaming():
            shared_keypoints.put(None)  # Add sentinel value to indicate end of stream
            break

# IR Thread

def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))

  data = np.fromiter(
     array_pointer.contents, dtype=np.dtype(np.uint16)
   ).reshape(
     frame.contents.height, frame.contents.width
   ) # copy

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
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

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

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

      try:
        while True:
          data = q.get(True, 500)
          if data is None:
            break
          data = cv2.resize(data[:,:], (960, 720))
          data = cv2.flip(data, 0)
          data = cv2.flip(data, 1)

          timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

          # Get the shared thermal ROI from the RGB thread
          try:
              shared_keypoint_data = shared_keypoints.get(block=False)
              nose = shared_keypoint_data["nose"]
              leye = shared_keypoint_data["leye"]
              reye = shared_keypoint_data["reye"]

              # Use the keypoints as needed
              if nose is not None:
                  # Perform actions using nose keypoint 
                  nose_x = int(nose.x-190)
                  nx1 = int(nose.x-130)
                  nx2 = int(nose.x-220)
                  nose_y = int(nose.y-20)
                  ny1 = int(nose.y+20)
                  ny2 = int(nose.y-30)
                  nose_loc = nose_x, nose_y
                  roi_data = data[ny2:ny1, nx2:nx1]
                  avgVal = np.mean(roi_data)
                  csv_writer.writerow([timestamp, avgVal])
                  pass
              if leye is not None:
                  # Perform actions using left eye keypoint 
                  leye_x = int(leye.x-180)
                  leye_y = int(leye.y-30)
                  leye_loc = leye_x, leye_y
                  lex1 = int(leye.x-160)
                  lex2 = int(leye.x-200)
                  ley1 = int(leye.y-10)
                  ley2 = int(leye.y-50)
                  le_data = data[ley2:ley1, lex2:lex1]
                  leMaxVal = np.max(le_data)              
                  pass
              if reye is not None:
                  # Perform actions using right eye keypoint 
                  reye_x = int(reye.x-180)
                  reye_y = int(reye.y-30)
                  reye_loc = reye_x, reye_y
                  rex1 = int(reye.x-160)
                  rex2 = int(reye.x-200)
                  rey1 = int(reye.y-10)
                  rey2 = int(reye.y-50)
                  re_data = data[rey2:rey1, rex2:rex1]
                  reMaxVal = np.max(re_data)
                  pass
           
          except queue.Empty:
              pass  # Queue is empty, continue without using keypoints

          img = raw_to_8bit(data)
          cv2.imshow('InfaSafe Thermal', img)
          cv2.waitKey(1)

        cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)


# Start the RGB/IR threads

ir_thread = threading.Thread(target=ir_camera_thread, daemon=True)
sensor_thread = threading.Thread(target=update_sensor_values, daemon=True)
rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)

ir_thread.start()
sensor_thread.start()
rgb_thread.start()


ir_thread.join()
sensor_thread.join()
rgb_thread.join()
