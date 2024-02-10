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
        self.buffer_size = buffer_size
        self.temperature_buffer = []
        self.humidity_buffer = []
        self.timestamp_buffer = []
        self.running = False

    def start(self):
        self.running = True
        update_thread = threading.Thread(target=self.update_sensor_values, daemon=True)
        update_thread.start()
 
    def update_sensor_values(self):
        global current_temperature, current_humidity
        while self.running:
            timestamp = datetime.datetime.now()
            current_temperature = self.sensor.temperature
            current_humidity = self.sensor.relative_humidity
          #  print(f"Timestamp: {timestamp}, Temp: {current_temperature}, Humnidity: {current_humidity}%")
            if len(self.temperature_buffer) >= self.buffer_size:
                self.temperature_buffer.pop(0)
                self.humidity_buffer.pop(0)
                self.timestamp_buffer.pop(0)  # Remove oldest data point
            self.temperature_buffer.append(current_temperature)
            self.humidity_buffer.append(current_humidity)
            self.timestamp_buffer.append(timestamp)
            time.sleep(1)

    def get_sensor_data(self):
        data = {
            'timestamps': [ts.isoformat() for ts in self.timestamp_buffer],
            'temperatures': self.temperature_buffer,
	    'humidity': self.humidity_buffer,
        }
        return data

def ktof(val):
    return (1.8 * ktoc(val) + 32.0)

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

    def start(self):
        self.running = True
        update_thread = threading.Thread(target=self.add_value, daemon=True)
        update_thread.start()


        
    def add_value(self, value):
        while self.running:
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
            self.add_value(value)
            time.sleep(1) 

            #print(f"Updated max_temperature: {self.max_temperature}")

        
    def export_baby_temp_data(self):
        # Export the data as a dictionary
        data = {
            'timestamps': [ts.isoformat() for ts in self.timestamp_buffer],
            'baby temperature': self.temperature_buffer,
        }
        return data

sensor_updater = SensorUpdater(sensor)
sensor_updater.start() 


app = Flask(__name__)

@app.route('/plot')
def get_sensor_data():
    data = sensor_updater.get_sensor_data()
    return jsonify(data)

@app.route('/baby_temp_plot')
def get_rr_data():
    baby_temp_data = sensor_updater_bt.export_baby_temp_data()
    return jsonify(baby_temp_data)

if __name__ == "__main__":
    app.run(host='10.0.0.209', port=5001, debug=True)
    

