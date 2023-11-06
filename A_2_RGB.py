#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
#import queue
#import time
#import numpy as np
#import cv2
#import logging
#import board
#import adafruit_ahtx0
#import datetime
#import scipy.signal
from jetson_inference import poseNet
import jetson_utils
#from uvctypes import *

# Constants
POSENET_MODEL = 'densenet121-body'
POSENET_THRESHOLD = 0.15
CAMERA_SOURCE = "csi://0"
DISPLAY_OUTPUT = "display://0"
#TEMP_HUMID_UPDATE_INTERVAL = 30  # seconds
#IR_CAMERA_VID = PT_USB_VID  # Replace with actual vendor ID
#IR_CAMERA_PID = PT_USB_PID  # Replace with actual product ID
#IR_CAMERA_FORMAT = UVC_FRAME_FORMAT_Y16
#IR_CAMERA_GUID = VS_FMT_GUID_Y16
#TEMP_CONVERSION_FACTOR = 27315
#TEMP_CONVERSION_OFFSET = 32.0
#TEMP_CONVERSION_SCALE = 100.0

# Configure logging
#logging.basicConfig(level=logging.DEBUG)

# AHT10 Temp/Humidity Sensor Setup
#i2c = board.I2C()  # uses board.SCL and board.SDA
#sensor = adafruit_ahtx0.AHTx0(i2c)

# Initialize global variables
#current_temperature = 0.0
#current_humidity = 0.0

# Queues
#BUF_SIZE = 2
#q = queue.Queue(BUF_SIZE)
#shared_keypoints = queue.Queue()
#plot_queue = queue.Queue()


# Class definitions
#class RespiratoryRateCalculator:
    # ... (Keep the class definition as is, but add logging and error handling)

class SensorUpdater:
    def __init__(self, sensor):
        self.sensor = sensor
        self.running = True

    def update_sensor_values(self):
        global current_temperature, current_humidity
        while self.running:
            current_temperature = self.sensor.temperature
            current_humidity = self.sensor.relative_humidity
            time.sleep(TEMP_HUMID_UPDATE_INTERVAL)

    def stop(self):
        self.running = False

class RGB_Camera:
    def __init__(self):
        self.running = True
        self.net = poseNet(POSENET_MODEL, threshold=POSENET_THRESHOLD)
        self.display = jetson_utils.videoOutput(DISPLAY_OUTPUT)

    def capture_frames(self):
        
        cam = jetson_utils.videoSource(CAMERA_SOURCE)

        while self.running:
            # Capture frame
            rgb_frame = cam.Capture()
            
            #if rgb_frame is None: # timeout
                #continue  
                
            # Perform pose estimation
            #poses = self.net.Process(rgb_frame)
            
            # Render image
            self.display.Render(rgb_frame)
            # Update display
            #self.display.SetStatus("PoseNet | Network {:.0f} FPS".format(self.net.GetNetworkFPS()))

            # Check for user input to stop the thread
            if not cam.IsStreaming() or not self.display.IsStreaming():
                self.stop()

    def stop(self):
        self.running = False
        logging.info("RGB_Camera stopped.")

    def run(self):
        try:
            self.capture_frames()
        except Exception as e:
            logging.error(f"An error occurred in RGB_Camera: {e}")
            self.stop()

#class IR_Camera:
    # ... (Refactor the ir_camera_thread function into this class)

# Main execution
if __name__ == "__main__":
    # Start the sensor updater thread
    #sensor_updater = SensorUpdater(sensor)
    #sensor_thread = threading.Thread(target=sensor_updater.update_sensor_values, daemon=True)
    #sensor_thread.start()

    # Start the RGB camera thread
    rgb_camera = RGB_Camera()
    rgb_thread = threading.Thread(target=rgb_camera.run, daemon=True)
    rgb_thread.start()

    # Start the IR camera thread
    #ir_camera = IR_Camera()
    #ir_thread = threading.Thread(target=ir_camera.run, daemon=True)
    #ir_thread.start()

    # Wait for threads to finish if necessary
    #try:
        #ir_thread.join()
        #sensor_thread.join()
        #rgb_thread.join()
    #except KeyboardInterrupt:
        #logging.info("Interrupt received, stopping threads.")
        #sensor_updater.stop()
        #rgb_camera.stop()
        #ir_camera.stop()
