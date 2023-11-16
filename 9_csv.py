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
import scipy.signal
import matplotlib.pyplot as plt
import csv

# Create a CSV file for data export
csv_filename = "body_temp_data.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Temp"])



# Create sensor object, communicating over the board's default I2C bus
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_ahtx0.AHTx0(i2c)

# Initialize variables to store temperature and humidity values
current_temperature = 0.0
current_humidity = 0.0

BUF_SIZE = 2
q = queue.Queue(BUF_SIZE)
shared_keypoints = queue.Queue()
plot_queue = queue.Queue()

class RespiratoryRateCalculator:
    def __init__(self, buffer_size=300, start_size=100):
        self.buffer_size = buffer_size
        self.start_size = start_size
        self.values_buffer = []
        self.filtered_buffer = []  # Store filtered data
        self.test_buffer = []
        self.normalized_buffer = []
        self.timestamps_buffer = []
        self.respiratory_rates = []  # Store calculated respiratory rates

    def add_value(self, value):
        timestamp = datetime.datetime.now()
        if len(self.values_buffer) >= self.buffer_size:
            self.values_buffer.pop(0)
            self.filtered_buffer.pop(0)  # Remove oldest filtered data point
            self.test_buffer.pop(0)
            self.normalized_buffer.pop(0)
            self.timestamps_buffer.pop(0)
        self.values_buffer.append(value)
        self.timestamps_buffer.append(timestamp)

    def calculate_rr(self):
        if len(self.values_buffer) < self.start_size:
            return None

        data = np.array(self.values_buffer)
        normalized_data = ( (data - np.mean(data)) / np.std(data) )
        self.normalized_buffer = normalized_data.tolist()
        
        b, a = scipy.signal.butter(2, [0.1, 0.85], btype='band')
        filtered_data = scipy.signal.lfilter(b, a, normalized_data)
        self.filtered_buffer= filtered_data.tolist()

        # Multiply the data by a Hamming window
        window = scipy.signal.hamming(len(filtered_data), sym=0)
        filtered_data *= window
        self.test_buffer= filtered_data.tolist()
        


        # FFT transform and modulus squared
        fft = np.fft.fft(filtered_data)
        fft_magnitude = np.absolute(fft)
        fft_power = np.square(fft_magnitude)
        
        fft_power[0] = 0

        # Frequency samples
        time_duration = (self.timestamps_buffer[-1] - self.timestamps_buffer[0]).total_seconds()
        sample_interval = time_duration / len(self.values_buffer)
        frequencies = np.fft.fftfreq(len(data), d=sample_interval)

        # Find the index of the maximum FFT value and get the respiration frequency
        max_idx = np.argmax(fft_power[1:]) + 1
        breaths_per_sec = frequencies[max_idx]
        breaths_per_min = breaths_per_sec * 60

        #print(f"FFT Magnitude: {fft_magnitude}")
        #print(f"FFT Power: {fft_power}")
        #print(f"Frequencies: {frequencies}")
        #print(f"Max FFT index: {max_idx}, Frequency: {frequencies[max_idx]}, Breaths per minute:{breaths_per_min}")

        # Store the calculated respiratory rate
        self.respiratory_rates.append(breaths_per_min)

        return breaths_per_min

    def reset(self):
        """Reset the buffers and respiratory rates."""
        self.values_buffer = []
        self.filtered_buffer = []  # Clear the filtered data
        self.normalized_buffer = []
        self.test_buffer = []
        self.respiratory_rates = []

    def plot_data(self, timestamps_to_plot):
        # Plot the raw data and filtered data on the same plot
        plt.figure(figsize=(12, 6))
        plt.subplot(4, 1, 1)  # One subplot
        plt.plot(timestamps_to_plot, self.values_buffer, label='Raw Data')
        plt.xlabel('Time')
        plt.ylabel('Temperature Value')
        plt.title('Raw Temperature Data Over Time')
        plt.legend()
        plt.subplot(4, 1, 2)  # One subplot
        plt.plot(timestamps_to_plot, self.normalized_buffer, label='normalized Data')
        plt.xlabel('Time')
        plt.ylabel('Normalized Value')
        plt.title('Normalized Data Over Time')
        plt.legend()
        plt.subplot(4, 1, 3)  # One subplot
        plt.plot(timestamps_to_plot, self.filtered_buffer, label='BW Filtered Data')
        plt.xlabel('Time')
        plt.ylabel('Temperature Value')
        plt.title('Filtered Temperature Data Over Time')
        plt.legend()
        plt.subplot(4, 1, 4)  # One subplot
        plt.plot(timestamps_to_plot, self.test_buffer, label='test Data')
        plt.xlabel('Time')
        plt.ylabel('Temperature Value')
        plt.title('Hamming Data Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()


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

            nose = pose.Keypoints[noseidx]
            
            if leyeidx > 0 and reyeidx > 0:
                reye = pose.Keypoints[reyeidx]
                leye = pose.Keypoints[leyeidx]
            elif reyeidx > 0:
                reye = pose.Keypoints[reyeidx]
            elif leyeidx > 0:
                leye = pose.Keypoints[leyeidx]
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
        
        
      roi_data = None
      ce_data = None 
      le_data = None 
      re_data = None

      try:
        while True:
          data = q.get(True, 500)
          if data is None:
            break
          data = cv2.resize(data[:,:], (960, 720))
          data = cv2.flip(data, 0)
          data = cv2.flip(data, 1)
          minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
          temp_timestamp = datetime.datetime.now()

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
                  nx1 = int(nose.x-120)
                  nx2 = int(nose.x-190)
                  nose_y = int(nose.y-20)
                  ny1 = int(nose.y-15)
                  ny2 = int(nose.y-40)
                  nose_loc = nose_x, nose_y
                  roi_data = data[ny2:ny1, nx2:nx1]
                  avgVal = np.mean(roi_data)
                  rr_calculator.add_value(avgVal)
                  # Periodically calculate the respiratory rate
                  if len(rr_calculator.values_buffer) >= rr_calculator.start_size:
                      rr = rr_calculator.calculate_rr()
                      if rr is not None:
                          print(f"Respiratory Rate: {rr:.2f} breaths/minute")
                          #print(rr_calculator.timestamps_buffer[-1])
                          #csv_writer.writerow([rr_calculator.timestamps_buffer[-1], rr])
                          # Instead of plotting here, put the data on the queue
                          plot_queue.put((rr_calculator.values_buffer, rr_calculator.timestamps_buffer))
              else:
                 print("no nose")
              if leye is not None and reye is not None:
                  # Combined actions for both eyes
                  cex1 = int(max(leye.x,reye.x)-130)
                  cex2 = int(min(leye.x,reye.x)-210)
                  cey1 = int(max(leye.y,reye.y)-30)
                  cey2 = int(min(leye.y,reye.y)-80)
                  ce_data = data[cey2:cey1, cex2:cex1]
                  ceMaxVal = ktof(np.max(ce_data))
                  csv_writer.writerow([temp_timestamp, ceMaxVal])
              elif leye is not None:
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
              elif reye is not None:
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
              else:
                  print("no eyes")
           
          except queue.Empty:
              pass  # Queue is empty, continue without using keypoints

          img = raw_to_8bit(data)


          if roi_data is not None:
              cv2.rectangle(img, (nx2, ny2), (nx1, ny1), (0, 255, 0), 2)  # Green rectangle
              display_temperature(img, avgVal, (nx1,ny1) , (0, 0, 255))
          if ce_data is not None:
              cv2.rectangle(img, (cex2, cey2), (cex1, cey1), (255, 0, 255), 2)  # Purple rectangle
              display_temperature(img, ceMaxVal, (cex1,cey1) , (0, 0, 255))
          if le_data is not None:
              cv2.rectangle(img, (lex2, ley2), (lex1, ley1), (0, 0, 255), 2)  # Red rectangle
          if re_data is not None:
             cv2.rectangle(img, (rex2, rey2), (rex1, rey1), (255, 0, 0), 2)  # Blue rectangle

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

# Create an instance of the RespiratoryRateCalculator
rr_calculator = RespiratoryRateCalculator()

ir_thread = threading.Thread(target=ir_camera_thread, daemon=True)
sensor_thread = threading.Thread(target=update_sensor_values, daemon=True)
rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)

ir_thread.start()
sensor_thread.start()
rgb_thread.start()

# Wait for threads to finish if necessary
ir_thread.join()
sensor_thread.join()
rgb_thread.join()
