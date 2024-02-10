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

IR_IMAGE_WIDTH = 960
IR_IMAGE_HEIGHT = 720

BUF_SIZE = 2
q = queue.Queue(BUF_SIZE)
shared_keypoints = queue.Queue()
plot_queue = queue.Queue()
health_plot_queue = queue.Queue()

class HealthMetricsCalculator:
    def __init__(self, buffer_size=100, fever_threshold=93.0, high_fever_threshold=96.5):
        self.buffer_size = buffer_size
        self.temperature_buffer = []
        self.timestamp_buffer = []
        self.max_temperature = None  # Initialize with None, will be set when values are added
        self.min_temperature = None  # Initialize with None, will be set when values are added
        self.fever_threshold = fever_threshold
        self.high_fever_threshold = high_fever_threshold
        self.high_fever_event = threading.Event()  # Event flag for critical fever
        self.fever_event = threading.Event()  # Event flag for fever
        
        
    def add_value(self, value):
        timestamp = datetime.datetime.now()
        bodyTemp = ktof(value)
        if len(self.temperature_buffer) >= self.buffer_size:
            self.temperature_buffer.pop(0)
            self.timestamp_buffer.pop(0)  # Remove oldest data point
        self.temperature_buffer.append(bodyTemp)
        self.timestamp_buffer.append(timestamp)
        #print(f"New body temperature: {bodyTemp}")
        #print(f"Current max_temperature: {self.max_temperature}")
        
        # Update the max and min temperatures
        if self.max_temperature is None or bodyTemp > self.max_temperature:
            self.max_temperature = bodyTemp
        if self.min_temperature is None or bodyTemp < self.min_temperature:
            self.min_temperature = bodyTemp
        
        if bodyTemp > self.high_fever_threshold:
                self.high_fever_event.set()
        elif bodyTemp > self.fever_threshold:
                self.fever_event.set()

        #print(f"Updated max_temperature: {self.max_temperature}")

        
    def export_data(self):
        # Export the data as a dictionary
        data = {
            'timestamps': self.timestamp_buffer,
            'temperatures': self.temperature_buffer,
        }
        return data
        

class RespiratoryRateCalculator:
    def __init__(self, health_metrics_calculator, high_RR_threshold=30, low_RR_threshold=3, buffer_size=270, start_size=120):
        self.buffer_size = buffer_size
        self.start_size = start_size
        self.values_buffer = []
        self.filtered_buffer = []
        self.filter_timestamps_buffer = []  # Store filtered data
        self.timestamps_buffer = []
        self.respiratory_rates = []  # Store calculated respiratory rates
        self.timestamps_rr_buffer = []
        self.fft_results_power = [] # Store FFT power results
        self.fft_frequencies = []  # Store FFT frequencies
        self.calculator = health_metrics_calculator
        self.low_RR_threshold = low_RR_threshold
        self.high_RR_threshold = high_RR_threshold
        self.high_RR_event = threading.Event()  
        self.low_RR_event = threading.Event() 

    def add_value(self, value):
        timestamp = datetime.datetime.now()
        if len(self.values_buffer) >= self.buffer_size:
            self.values_buffer.pop(0)
            self.timestamps_buffer.pop(0)  # Remove oldest data point
        self.values_buffer.append(value)
        self.timestamps_buffer.append(timestamp)

    def hampel_filter(self, input_series, window_size, n_sigmas=3):
        new_series = input_series.copy()
        L = 1.4826  # scale factor for Gaussian distribution
        rolling_median = np.array([np.median(input_series[max(i - window_size // 2, 0):min(i + window_size // 2, len(input_series))]) for i in range(len(input_series))])
        MAD = np.array([np.median(np.abs(input_series[max(i - window_size // 2, 0):min(i + window_size // 2, len(input_series))] - rolling_median[i])) for i in range(len(input_series))])
        threshold = n_sigmas * L * MAD
        outlier_idx = np.abs(input_series - rolling_median) > threshold
        new_series[outlier_idx] = rolling_median[outlier_idx]
        return new_series


    def calculate_rr(self):
        if len(self.values_buffer) < self.start_size:
            return None

        data = np.array(self.values_buffer)
        tstamp_filter = np.array(self.timestamps_buffer)
        normalized_data = (data - np.mean(data)) / np.std(data)
        
        # samples
        sample_interval = 1 / 9
        
        # Convert bpm to Hz for the filter
        low_freq = 0.015  # 1 bpm in Hz
        high_freq = 0.85  # 50 bpm in Hz

        # Normalize the frequencies by the Nyquist frequency (half the sampling rate)
        nyquist = 9 / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        hampel_data = self.hampel_filter(normalized_data, window_size=50)

        # Apply the bandpass Butterworth filter
        b, a = scipy.signal.butter(2, [low, high], btype='band')    
        filtered_data = scipy.signal.filtfilt(b, a, hampel_data)
        self.filtered_buffer = filtered_data.tolist()
        self.filter_timestamps_buffer = tstamp_filter.tolist()
        
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

        # Check if the temperature exceeds the threshold and set the event if it does
        if breaths_per_min > self.high_RR_threshold:
            self.high_RR_event.set()
        if breaths_per_min < self.low_RR_threshold:
            self.low_RR_event.set()

        # Store the calculated respiratory rate
        if len(self.timestamps_rr_buffer) >= self.buffer_size:
            self.timestamps_rr_buffer.pop(0)
            self.respiratory_rates.pop(0)
        self.respiratory_rates.append(breaths_per_min)
        self.timestamps_rr_buffer.append(self.timestamps_buffer[-1])

        return breaths_per_min

    def reset(self):
        """Reset the buffers and respiratory rates."""
        self.values_buffer = []
        self.filtered_buffer = []  # Clear the filtered data
        self.respiratory_rates = []
        self.timestamps_buffer = [] 
        self.filter_timestamps_buffer = []
        self.timestamps_rr_buffer = []
        self.fft_results_power = []
        self.fft_frequencies = []  # Store FFT frequencies

    def plot_data(self):
        # Plot the raw data and filtered data on the same plot

        if len(self.timestamps_rr_buffer) != len(self.respiratory_rates):
            print(f"Cannot plot data. Lengths do not match: timestamps ({len(self.timestamps_buffer)}) vs respiratory rates ({len(self.respiratory_rates)})")
            return  # Exit the function if lengths do not match

        fft_power = self.fft_results_power[-1]
        frequencies = self.fft_frequencies[-1]
        
        
        plt.figure(figsize=(12, 6))

        # Normalize raw data for comparison (optional)
        normalized_data = (self.values_buffer - np.mean(self.values_buffer)) / np.std(self.values_buffer)
        plt.plot(self.timestamps_buffer, normalized_data, label='Normalized Data', alpha=0.7)

        # Plot Hampel filtered data
        hampel_data = self.hampel_filter(np.array(normalized_data), window_size=50)
        plt.plot(self.filter_timestamps_buffer, hampel_data, label='Hampel Filtered Data', alpha=0.7)

        # Plot Butterworth filtered data
        plt.plot(self.filter_timestamps_buffer, self.filtered_buffer, label='Butterworth Filtered Data', alpha=0.7)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Data Processing Stages')
        plt.legend()
        plt.show()
        
        exported_body_data = self.calculator.export_data()
        if exported_body_data is not None:    
            body_temp_timestamps = exported_body_data['timestamps']
            body_temperatures = exported_body_data['temperatures']
            
            max_temperature = max(body_temperatures)
            min_temperature = min(body_temperatures)
            max_temp_index = body_temperatures.index(max_temperature)
            min_temp_index = body_temperatures.index(min_temperature)
            
            plt.subplot(5, 1, 5)  # One subplot
            plt.plot(body_temp_timestamps, body_temperatures, label='Temperature Data')
            
            # Plot max temperature
            plt.plot(body_temp_timestamps[max_temp_index], max_temperature, 'ro', label='Max Temperature')
            plt.annotate(f'Max: {max_temperature}°F', 
                         (body_temp_timestamps[max_temp_index], max_temperature), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center')

            # Plot min temperature
            plt.plot(body_temp_timestamps[min_temp_index], min_temperature, 'bo', label='Min Temperature')
            plt.annotate(f'Min: {min_temperature}°F', 
                         (body_temp_timestamps[min_temp_index], min_temperature), 
                         textcoords="offset points", 
                         xytext=(0,-15), 
                         ha='center')

            plt.xlabel('Time')
            plt.ylabel('Temperature (°F)')
            plt.title('Body Temperature Data Over Time')
            plt.legend()
            
        plt.tight_layout()
        plt.show()


# Sensor Class to update temperature and humidity values
class SensorUpdater:
    def __init__(self, sensor, update_interval, high_room_temp_threshold, high_room_rh_threshold):
        self.sensor = sensor
        self.running = True
        self.update_interval = update_interval
        self.high_room_temp_threshold = high_room_temp_threshold  # High temperature threshold
        self.high_room_rh_threshold = high_room_rh_threshold  # High temperature threshold
        self.high_room_temp_event = threading.Event()  # Event flag for high temperature
        self.high_room_rh_event = threading.Event()  # Event flag for high temperature


    def update_sensor_values(self):
        global current_temperature, current_humidity
        while self.running:
            current_temperature = round(self.sensor.temperature, 2)
            current_humidity = round(self.sensor.relative_humidity, 2)
            print("\nTemperature: %0.1f C" % current_temperature)
            print("Humidity: %0.1f %%" % current_humidity)

            # Check if the temperature exceeds the threshold and set the event if it does
            if current_temperature > self.high_room_temp_threshold:
                self.high_room_temp_event.set()
            if current_humidity > self.high_room_rh_threshold:
                self.high_room_rh_event.set()

            time.sleep(self.update_interval)

    def stop(self):
        self.running = False

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
                event_monitor.set_event(no_nose_event_name)
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
        font.OverlayText(rgb_frame, text=f"Temp: {current_temperature:.1f} C   Humidity: {current_humidity:.1f} %", x=5, y= 5 + (font.GetSize() + 5), color=font.White, background=font.Gray40)

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
  
def is_within_boundaries(x, y, width, height):
    # Check if the coordinates are within the image boundaries
    return 0 <= x < width and 0 <= y < height
    
def is_region_within_boundaries(x1, y1, x2, y2, width, height):
    # Check if both top-left and bottom-right corners are within the image boundaries
    return (0 <= x1 < width) and (0 <= y1 < height) and (0 <= x2 < width) and (0 <= y2 < height)

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
      nose_region_in_bounds = False
      ceye_region_in_bounds = False
      leye_region_in_bounds = False
      reye_region_in_bounds = False
      x_shift = 150
      y_shift = 70
      x_width = 30
      y_height = 20

      try:
        while True:
          data = q.get(True, 500)
          if data is None:
            break
          data = cv2.resize(data[:,:], (IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT))
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
                  nose_x = int(nose.x-x_shift)
                  nose_y = int(nose.y+y_shift)
                  nx1 = int(nose_x+x_width)
                  ny1 = int(nose_y+x_width)
                  nx2 = int(nose_x-x_width)
                  ny2 = int(nose_y)
                  nose_loc = nose_x, nose_y
                  nose_region_in_bounds = is_region_within_boundaries(nx1, ny1, nx2, ny2, IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT)
                  if nose_region_in_bounds:
                     roi_data = data[ny2:ny1, nx2:nx1]
                     avgVal = np.mean(roi_data)
                     rr_calculator.add_value(avgVal)
                     # Periodically calculate the respiratory rate
                     if len(rr_calculator.values_buffer) >= rr_calculator.start_size:
                        rr = rr_calculator.calculate_rr()
                        if rr is not None:
                            #print(f"Respiratory Rate: {rr:.2f} breaths/minute")
                            # Instead of plotting here, put the data on the queue
                            plot_queue.put((rr_calculator.values_buffer))
                  else:
                    # Set flag or log that the nose region is out of bounds
                    event_monitor.set_event(out_of_bounds_event_name)
                    # ... code to handle out of bounds ...
                    continue  # Skip further processing for this keypoint
              else:
                  rr_calculator.reset()
              if leye is not None and reye is not None:
                  # Combined actions for both eyes
                  leye_x = int(leye.x-x_shift)
                  leye_y = int(leye.y+y_shift)
                  leye_loc = leye_x, leye_y
                  reye_x = int(reye.x-x_shift)
                  reye_y = int(reye.y+y_shift)
                  reye_loc = reye_x, reye_y
                  cex1 = int(max(leye_x,reye_x)+x_width)
                  cex2 = int(min(leye_x,reye_x)-x_width)
                  cey1 = int(max(leye_y,reye_y)+(y_height))
                  cey2 = int(min(leye_y,reye_y)-(y_height))
                  ceye_region_in_bounds = is_region_within_boundaries(cex1, cey1, cex2, cey2, IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT)
                  if ceye_region_in_bounds:
                      ce_data = data[cey2:cey1, cex2:cex1]
                      ceMaxVal = np.max(ce_data)
                      health_calc.add_value(ceMaxVal)
                      health_plot_queue.put((health_calc.temperature_buffer))
                      pass
                  else:
                    # Set flag or log that the nose region is out of bounds
                    event_monitor.set_event(out_of_bounds_event_name)
                    # ... code to handle out of bounds ...
                    # Skip further processing for this keypoint
                    continue
              elif leye is not None:
                  # Perform actions using left eye keypoint 
                  leye_x = int(leye.x-x_shift)
                  leye_y = int(leye.y+y_shift)
                  leye_loc = leye_x, leye_y
                  lex1 = int(leye_x+(x_width/2))
                  lex2 = int(leye_x-(x_width/2))
                  ley1 = int(leye_y+(y_height))
                  ley2 = int(leye_y-(y_height))
                  leye_region_in_bounds = is_region_within_boundaries(lex1, ley1, lex2, ley2, IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT)
                  if leye_region_in_bounds:
                      le_data = data[ley2:ley1, lex2:lex1]
                      leMaxVal = np.max(le_data)
                      health_calc.add_value(ktof(leMaxVal))
                      health_plot_queue.put((health_calc.temperature_buffer))                  
                      pass
                  else:
                    # Set flag or log that the nose region is out of bounds
                    event_monitor.set_event(out_of_bounds_event_name)
                    # ... code to handle out of bounds ...
                    # Skip further processing for this keypoint
                    continue
              elif reye is not None:
                  # Perform actions using right eye keypoint 
                  reye_x = int(reye.x-x_shift)
                  reye_y = int(reye.y+y_shift)
                  reye_loc = reye_x, reye_y
                  rex1 = int(reye_x+(x_width/2))
                  rex2 = int(reye_x-(x_width/2))
                  rey1 = int(reye_y+(y_height))
                  rey2 = int(reye_y-(y_height))
                  reye_region_in_bounds = is_region_within_boundaries(rex1, rey1, rex2, rey2, IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT)
                  if reye_region_in_bounds:
                      re_data = data[rey2:rey1, rex2:rex1]
                      reMaxVal = np.max(re_data)
                      health_calc.add_value(ktof(reMaxVal))
                      health_plot_queue.put((health_calc.temperature_buffer))
                      pass
                  else:
                    # Set flag or log that the nose region is out of bounds
                    event_monitor.set_event(out_of_bounds_event_name)
                    # ... code to handle out of bounds ...
                    # Skip further processing for this keypoint
                    continue
              else:
                  print("no eyes")
           
          except queue.Empty:
              pass  # Queue is empty, continue without using keypoints

          img = raw_to_8bit(data)


          if nose_region_in_bounds:
              cv2.circle(img, nose_loc, 5, (0, 255, 0), 5)
              cv2.rectangle(img, (nx2, ny2), (nx1, ny1), (0, 255, 0), 2)  # Green rectangle
              display_temperature(img, avgVal, (nx1,ny1) , (0, 0, 255))
          if ceye_region_in_bounds:
              cv2.circle(img, leye_loc, 5, (0, 0, 255), 5)
              cv2.circle(img, reye_loc, 5, (255, 0, 0), 5)
              cv2.rectangle(img, (cex2, cey2), (cex1, cey1), (255, 0, 255), 2)  # Purple rectangle
              display_temperature(img, ceMaxVal, (cex1,cey1) , (0, 0, 255))
          if leye_region_in_bounds:
              cv2.circle(img, leye_loc, 5, (0, 0, 255), 5)
              cv2.rectangle(img, (lex2, ley2), (lex1, ley1), (0, 0, 255), 2)  # Red rectangle
          if reye_region_in_bounds:
             cv2.circle(img, reye_loc, 5, (255, 0, 0), 5)
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
            time.sleep(5)  # Sleep to prevent busy waiting

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.running = False

def event_processing_thread():
    while True:
        if sensor_updater.high_room_temp_event.is_set():
            event_monitor.set_event(high_room_temp_event_name)
            sensor_updater.high_room_temp_event.clear()  # Reset the event

        if sensor_updater.high_room_rh_event.is_set():
            event_monitor.set_event(high_room_rh_event_name)
            sensor_updater.high_room_rh_event.clear()  # Reset the event

        # Check for out of bounds event
        if event_monitor.events[out_of_bounds_event_name].is_set():
            event_monitor.set_event(out_of_bounds_event_name)
            event_monitor.reset_event(out_of_bounds_event_name)

        # Check for unsafe sleeping position event
        if event_monitor.events[no_nose_event_name].is_set():
            event_monitor.set_event(no_nose_event_name)
            event_monitor.reset_event(no_nose_event_name)

        # Check for fever event
        if health_calc.fever_event.is_set():
            event_monitor.set_event(fever_event_name)
            health_calc.fever_event.clear()  # Reset the event

        if health_calc.high_fever_event.is_set():
            event_monitor.set_event(high_fever_event_name)
            health_calc.high_fever_event.clear()  # Reset the event

        # Check for RR event
        if rr_calculator.high_RR_event.is_set():
            event_monitor.set_event(high_RR_event_name)
            rr_calculator.high_RR_event.clear()  # Reset the event

        if rr_calculator.low_RR_event.is_set():
            event_monitor.set_event(low_RR_event_name)
            rr_calculator.low_RR_event.clear()  # Reset the event

        time.sleep(0.1)



event_monitor = EventMonitor()

# Create an instance of the RespiratoryRateCalculator
health_calc = HealthMetricsCalculator()
rr_calculator = RespiratoryRateCalculator(health_calc)

sensor_updater = SensorUpdater(sensor=sensor, update_interval=30, high_room_temp_threshold=23, high_room_rh_threshold=40)

# Register events with the EventMonitor
high_room_temp_event_name = 'High Room Temperature Event'
event_monitor.register_event(high_room_temp_event_name)

high_room_rh_event_name = 'High Room Relative Humidity Event'
event_monitor.register_event(high_room_rh_event_name)

out_of_bounds_event_name = 'ROI Out of Bounds Event'
event_monitor.register_event(out_of_bounds_event_name)

no_nose_event_name = 'Unsafe Sleeping Position Event'
event_monitor.register_event(no_nose_event_name)

fever_event_name = 'Fever Detected Event'
event_monitor.register_event(fever_event_name)

high_fever_event_name = 'High Fever Detected Event'
event_monitor.register_event(high_fever_event_name)

high_RR_event_name = 'High Breathing Rate Detected Event'
event_monitor.register_event(high_RR_event_name)

low_RR_event_name = 'Low Breathing Rate Detected Event'
event_monitor.register_event(low_RR_event_name)

# Start the RGB/IR threads
ir_thread = threading.Thread(target=ir_camera_thread, daemon=True)
sensor_thread = threading.Thread(target=sensor_updater.update_sensor_values, daemon=True)
rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)
monitor_thread = threading.Thread(target=event_monitor.monitor_events, daemon=True)
event_processing_thread = threading.Thread(target=event_processing_thread, daemon=True)

monitor_thread.start()
event_processing_thread.start()
ir_thread.start()
sensor_thread.start()
rgb_thread.start()

plot_update_interval = 5  # seconds

try:
    last_plot_time = time.time()
    while True:
        current_time = time.time()
        # Update the plot if the interval has passed
        if current_time - last_plot_time > plot_update_interval:
            try:
                rr_to_plot = plot_queue.get_nowait()
                rr_calculator.plot_data()
                last_plot_time = current_time
            except queue.Empty:
                pass

        time.sleep(0.1)  # Sleep to prevent this loop from hogging the CPU

except KeyboardInterrupt:
    print("Exiting...")
    sensor_updater.stop()
    stop_signal.set()
    # Perform any cleanup here
finally:
    # Wait for threads to finish if necessary
    event_monitor.stop_monitoring()
    event_processing_thread.join()
    monitor_thread.join()
    ir_thread.join()
    sensor_thread.join()
    rgb_thread.join()



