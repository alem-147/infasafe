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

# Global flag to signal thread to stop
stop_signal = threading.Event()

# Create sensor object, communicating over the board's default I2C bus
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_ahtx0.AHTx0(i2c)

# Initialize variables to store temperature and humidity values
current_temperature = 0.0
current_humidity = 0.0
body_temp = 0.0

BUF_SIZE = 2
q = queue.Queue(BUF_SIZE)
shared_keypoints = queue.Queue()
plot_queue = queue.Queue()

class RespiratoryRateCalculator:
    def __init__(self, buffer_size=270, start_size=120):
        self.buffer_size = buffer_size
        self.start_size = start_size
        self.values_buffer = []
        self.filtered_buffer = []
        self.timestamps_buffer = []  # Store filtered data
        self.respiratory_rates = []  # Store calculated respiratory rates
        self.timestamps_rr_buffer = []
        self.fft_results_power = [] # Store FFT power results
        self.fft_frequencies = []  # Store FFT frequencies

    def add_value(self, value):
        timestamp = datetime.datetime.now()
        if len(self.values_buffer) >= self.buffer_size:
            self.values_buffer.pop(0)
            self.filtered_buffer.pop(0)
            self.timestamps_buffer.pop(0)  # Remove oldest data point
        self.values_buffer.append(value)
        self.timestamps_buffer.append(timestamp)


    def calculate_rr(self):
        if len(self.values_buffer) < self.start_size:
            return None

        data = np.array(self.values_buffer)
        
        # samples
        sample_interval = 1 / 9
        
        # Convert bpm to Hz for the filter
        low_freq = 10 / 60  # 10 bpm in Hz
        high_freq = 100 / 60  # 100 bpm in Hz

        # Normalize the frequencies by the Nyquist frequency (half the sampling rate)
        nyquist = 9 / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Apply the bandpass Butterworth filter
        b, a = scipy.signal.butter(2, [low, high], btype='band')    
        filtered_data = scipy.signal.filtfilt(b, a, data)
        self.filtered_buffer= filtered_data.tolist()
        
        # FFT transform and modulus squared
        fft = np.fft.fft(filtered_data)
        fft_magnitude = np.absolute(fft)
        fft_power = np.square(fft_magnitude)
        
        fft_power[0] = 0
        
        # Frequency samples
        frequencies = np.fft.fftfreq(len(data), d=sample_interval)
        
        self.fft_results_power.append(fft_power)
        self.fft_frequencies.append(frequencies)

        # Find the index of the maximum FFT value and get the respiration frequency
        max_idx = np.argmax(fft_power)
        breaths_per_sec = frequencies[max_idx]
        breaths_per_min = abs(breaths_per_sec * 60)

        # Store the calculated respiratory rate
        self.respiratory_rates.append(breaths_per_min)
        self.timestamps_rr_buffer.append(self.timestamps_buffer[-1])

        return breaths_per_min

    def reset(self):
        """Reset the buffers and respiratory rates."""
        self.values_buffer = []
        self.filtered_buffer = []  # Clear the filtered data
        self.respiratory_rates = []
        self.rr_timestamps_buffer = []
        self.fft_results_power = []
        self.fft_frequencies = []  # Store FFT frequencies

    def plot_data(self):
        # Plot the raw data and filtered data on the same plot

        if len(self.timestamps_rr_buffer) != len(self.respiratory_rates):
            print(f"Cannot plot data. Lengths do not match: timestamps ({len(self.timestamps_buffer)}) vs respiratory rates ({len(self.respiratory_rates)})")
            return  # Exit the function if lengths do not match
        
        fft_power = self.fft_results_power[-1]
        frequencies = self.fft_frequencies[-1]
        
        plt.figure(figsize=(12, 8))
        plt.subplot(5, 1, 1)  # One subplot
        plt.plot(self.timestamps_buffer, self.values_buffer, label='Raw Data')
        plt.xlabel('Time')
        plt.ylabel('Temperature Value')
        plt.title('Raw Temperature Data Over Time')
        plt.legend()
        plt.subplot(5, 1, 2)  # One subplot
        plt.plot(self.timestamps_buffer, self.filtered_buffer, label='Filtered Raw Data')
        plt.xlabel('Time')
        plt.ylabel('Filtered Value')
        plt.title('Butterworth Filtered Data Over Time')
        plt.legend()
        plt.subplot(5, 1, 3)  # One subplot
        plt.plot(frequencies[:len(frequencies)//2], fft_power[:len(fft_power)//2], label='FFT Power Output')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('FFT Power of Data')
        plt.legend()
        plt.subplot(5, 1, 4)  # One subplot
        plt.plot(self.timestamps_rr_buffer, self.respiratory_rates, label='Breathing Data')
        plt.xlabel('Time')
        plt.ylabel('Respiration Rate Value')
        plt.title('Respiration Rate Data Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()


# RGB Thread functions

def rgb_camera_thread():

    # Load the pose estimation model
    net = poseNet('densenet121-body', threshold=0.15)

    # Initialize the camera or video input source
    cam = jetson_utils.videoSource("csi://0")
    disp = jetson_utils.videoOutput("display://0")

    # Create a font for text overlay
    font = jetson_utils.cudaFont()

    while not stop_signal.is_set():  # Check if the stop signal has been set

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
                          # Instead of plotting here, put the data on the queue
                          plot_queue.put((rr_calculator.values_buffer))
              else:
                 print("no nose")
              if leye is not None and reye is not None:
                  # Combined actions for both eyes
                  cex1 = int(max(leye.x,reye.x)-130)
                  cex2 = int(min(leye.x,reye.x)-210)
                  cey1 = int(max(leye.y,reye.y)-30)
                  cey2 = int(min(leye.y,reye.y)-80)
                  ce_data = data[cey2:cey1, cex2:cex1]
                  ceMaxVal = np.max(ce_data)
                  body_temp_updater.check_high_body_temperature(ceMaxVal)
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
    
# Sensor Class to update temperature and humidity values
class SensorUpdater:
    def __init__(self, sensor, update_interval, high_temp_threshold):
        self.sensor = sensor
        self.running = True
        self.update_interval = update_interval
        self.high_temp_threshold = high_temp_threshold  # High temperature threshold
        self.high_room_temp_event = threading.Event()  # Event flag for high temperature


    def update_sensor_values(self):
        global current_temperature, current_humidity
        while self.running:
            current_temperature = round(self.sensor.temperature, 2)
            current_humidity = round(self.sensor.relative_humidity, 2)
            print(current_temperature)

            # Check if the temperature exceeds the threshold and set the event if it does
            if current_temperature > self.high_temp_threshold:
                self.high_room_temp_event.set()

            time.sleep(self.update_interval)

    def stop(self):
        self.running = False
    
    
class BodyTemperatureUpdater:
    def __init__(self, threshold):
        self.threshold = threshold
        self.high_body_temp_event = threading.Event()

    def update_body_temperature(self, max_val):
        global body_temp
        body_temp = ktof(max_val)  # You can adjust this calculation based on your requirements
        return body_temp

    def check_high_body_temperature(self, max_val):
        body_temp = self.update_body_temperature(max_val)
        if body_temp > self.threshold:
            self.high_body_temp_event.set()

    def reset_high_room_temp_event(self):
        self.high_body_temp_event.clear()

class EventMonitor:
    def __init__(self):
        self.events = {}
        self.running = True

    def register_event(self, event_name):
        """Register a new event with a given name."""
        event = threading.Event()
        self.events[event_name] = event
        return event

    def set_event(self, event_name):
        """Set the event to true, indicating it has occurred."""
        if event_name in self.events:
            self.events[event_name].set()

    def reset_event(self, event_name):
        """Reset the event to false after acknowledging it."""
        if event_name in self.events:
            self.events[event_name].clear()

    def monitor_events(self):
        """Monitor for any events that are set and print them."""
        while self.running:
            for event_name, event in self.events.items():
                if event.is_set():
                    print(f"Event occurred: {event_name}")
                    self.reset_event(event_name)
            time.sleep(0.1)  # Sleep to prevent busy waiting

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.running = False


# Start the RGB/IR threads

event_monitor = EventMonitor()

# Create an instance of the RespiratoryRateCalculator
rr_calculator = RespiratoryRateCalculator()

sensor_updater = SensorUpdater(sensor=sensor, update_interval=2, high_temp_threshold=23)

# Register events with the EventMonitor
high_room_temp_event_name = 'High Room Temperature Event'
event_monitor.register_event(high_room_temp_event_name)

high_body_temp_event_name = 'High Body Temperature Event'
event_monitor.register_event(high_body_temp_event_name)

body_temp_updater = BodyTemperatureUpdater(threshold=92)

ir_thread = threading.Thread(target=ir_camera_thread, daemon=True)
sensor_thread = threading.Thread(target=sensor_updater.update_sensor_values, daemon=True)
rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)

monitor_thread = threading.Thread(target=event_monitor.monitor_events, daemon=True)
monitor_thread.start()

ir_thread.start()
sensor_thread.start()
rgb_thread.start()

try:
    while True:
        # Check if there is new data to plot
        #try:
            # Get data from the queue without blocking
            #data_to_plot = plot_queue.get_nowait()
        if sensor_updater.high_room_temp_event.is_set():
           event_monitor.set_event(high_room_temp_event_name)
           sensor_updater.high_room_temp_event.clear()  # Reset the event
        elif body_temp_updater.high_body_temp_event.is_set():
           event_monitor.set_event(high_body_temp_event_name)
           body_temp_calculator.reset_high_body_temp_event()
            # Plot the data using the main thread
            #rr_calculator.plot_data()
        #except queue.Empty:
            # No data to plot, can sleep or do other work
           time.sleep(2)
except KeyboardInterrupt:
    print("Exiting...")
    sensor_updater.stop()
    stop_signal.set()
    # Perform any cleanup here
finally:
    # Wait for threads to finish if necessary
    event_monitor.stop_monitoring()
    monitor_thread.join()
    ir_thread.join()
    sensor_thread.join()
    rgb_thread.join()



