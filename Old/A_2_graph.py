#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import queue
import time
import logging
import board
import adafruit_ahtx0
import datetime
from flask import Flask, render_template, jsonify
import matplotlib.pyplot as pit
from io import BytesIO
import base64
import json

TEMP_HUMID_UPDATE_INTERVAL = 1/1000  # seconds


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# AHT10 Temp/Humidity Sensor Setup
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_ahtx0.AHTx0(i2c)

# Initialize global variables
current_temperature = 0.0
current_humidity = 0.0



class SensorUpdater:
    def __init__(self, sensor, buffer_size=100):
        self.sensor = sensor
        self.running = True
        self.buffer_size = buffer_size
        self.temperature_buffer = []
        self.humidity_buffer = []
        self.timestamp_buffer = []

    def update_sensor_values(self):
        global current_temperature, current_humidity
        while self.running:
            timestamp = datetime.datetime.now()
            current_temperature = self.sensor.temperature
            current_humidity = self.sensor.relative_humidity
            if len(self.temperature_buffer) >= self.buffer_size:
                self.temperature_buffer.pop(0)
                self.timestamp_buffer.pop(0)  # Remove oldest data point
            self.temperature_buffer.append(current_temperature)
            self.timestamp_buffer.append(timestamp)
            print(current_temperature,current_humidity)
            time.sleep(TEMP_HUMID_UPDATE_INTERVAL)

    def stop(self):
        self.running = False
    
    app = Flask(__name__)
    @app.route('/plot')    
    def export_data(self):
        # Export the data as a dictionary
        data = {
            'timestamps': [ts.isoformat() for ts in self.timestamp_buffer],
            'rates': self.temperature_buffer,
        }
        return jsonify(data)


# Main execution
if __name__ == "__main__":
    app.run(host='10.0.0.209', port=5001,  debug=True)

    # Start the sensor updater thread
    sensor_updater = SensorUpdater(sensor)
    sensor_thread = threading.Thread(target=sensor_updater.update_sensor_values, daemon=True)
    sensor_thread.start()
    sensor_thread.join()
 


                   


