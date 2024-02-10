import threading
import time
from collections import deque
import board
import adafruit_ahtx0
from flask import Flask, jsonify
from flask_cors import CORS

from jetson_inference import poseNet
import jetson_utils

ROOM_LOW_TEMP_THRESHOLD = 60.0
ROOM_HIGH_TEMP_THRESHOLD = 70.0
ROOM_LOW_RH_THRESHOLD = 40.0
ROOM_HIGH_RH_THRESHOLD = 50.0

CAM_POSENET = '/dev/video0'
DISP_POSENET = 'rtsp://10.0.0.209:5005/live' # 'webrtc://10.0.0.209:8554/infa_live'

# Initialize global variables for sensor readings
current_temperature = 0.0
current_humidity = 0.0

# Initialize the stop signal for threads
stop_signal = threading.Event()


# Simplified DataBuffer with external timestamp support and min/max tracking
class DataBuffer:
    def __init__(self, duration=60):
        self.buffer = deque()
        self.duration = duration
        self.min_value = None
        self.min_timestamp = None
        self.max_value = None
        self.max_timestamp = None

    def add_data(self, timestamp, value):
        self.buffer.append((timestamp, value))
        self.update_min_max_on_add(timestamp, value)
        self.trim_buffer(timestamp)

    def update_min_max_on_add(self, timestamp, value):
        if self.min_value is None or value < self.min_value:
            self.min_value = value
            self.min_timestamp = timestamp
        if self.max_value is None or value > self.max_value:
            self.max_value = value
            self.max_timestamp = timestamp

    def trim_buffer(self, current_time):
        while self.buffer and current_time - self.buffer[0][0] > self.duration:
            _, value = self.buffer.popleft()
            self.recalculate_min_max()

    def recalculate_min_max(self):
        # Reset min and max values
        self.min_value = None
        self.max_value = None
        self.min_timestamp = None
        self.max_timestamp = None
        for timestamp, value in self.buffer:
            if self.min_value is None or value < self.min_value:
                self.min_value = value
                self.min_timestamp = timestamp
            if self.max_value is None or value > self.max_value:
                self.max_value = value
                self.max_timestamp = timestamp

    def get_data(self):
        return list(self.buffer)
        
    def get_min_max(self):
        return (self.min_value, self.min_timestamp), (self.max_value, self.max_timestamp)

# AlertManager to evaluate data and generate alerts
class AlertManager:
    def __init__(self):
        self.alerts = []

    def check_for_alerts(self, data_type, value, timestamp):
        if data_type == 'temperature':
            if value > ROOM_HIGH_TEMP_THRESHOLD:  # Example condition
                self.alerts.append((timestamp, 'High Room Temperature Alert'))
                pass
            elif value < ROOM_LOW_TEMP_THRESHOLD:  # Example condition
                self.alerts.append((timestamp, 'High Room Temperature Alert'))
                pass
        if data_type == 'humidity':
            if value > ROOM_HIGH_RH_THRESHOLD:  # Example condition
                self.alerts.append((timestamp, 'High Room Humidity Alert'))
                pass
            elif value < ROOM_LOW_RH_THRESHOLD:  # Example condition
                self.alerts.append((timestamp, 'Low Room Humidity Alert'))
                pass
        # Add more conditions as needed

    def get_alerts(self):
        return self.alerts

# Modified SensorUpdater to use a central timestamp
class SensorUpdater:
    def __init__(self, sensor, update_interval, alert_manager=None):
        self.sensor = sensor
        self.update_interval = update_interval
        self.running = True
        self.alert_manager = alert_manager

    def update_sensor_values(self):
        global current_temperature, current_humidity
        while self.running:
            timestamp = time.time()
            current_temperature = round(((self.sensor.temperature*9/5)+32), 2)
            current_humidity = round(self.sensor.relative_humidity, 2)
            #print(current_temperature)
            #print(current_humidity)
            # Example: Update temperature and humidity buffers and check for alerts
            temperature_buffer.add_data(timestamp, current_temperature)
            humidity_buffer.add_data(timestamp, current_humidity)
            if self.alert_manager:
                self.alert_manager.check_for_alerts('temperature', current_temperature, timestamp)
                self.alert_manager.check_for_alerts('humidity', current_humidity, timestamp)
            time.sleep(self.update_interval)

    def stop(self):
        self.running = False

def rgb_camera_thread():

    # Load the pose estimation model
    net = poseNet('densenet121-body', threshold=0.15)

    # Initialize the camera or video input source
    cam = jetson_utils.videoSource(CAM_POSENET)
    disp = jetson_utils.videoOutput(DISP_POSENET)

    # Create a font for text overlay
    font = jetson_utils.cudaFont()

    while not stop_signal.is_set():  # Check if the stop signal has been set

        # Capture the next image from the RGB camera
        rgb_frame = cam.Capture()

        if rgb_frame is None:
            continue

        # Perform pose estimation on the RGB frame
        poses = net.Process(rgb_frame)

        # Overlay temperature and humidity values on the frame
        font.OverlayText(rgb_frame, text=f"Temp: {current_temperature:.1f} C   Humidity: {current_humidity:.1f} %", x=5, y= 5 + (font.GetSize() + 5), color=font.White, background=font.Gray40)

        # Display the RGB frame with overlays
        disp.Render(rgb_frame)

        # Update the output window
        disp.SetStatus("Pose Estimation | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        # Check for exit condition
        if not disp.IsStreaming() or not cam.IsStreaming():
            break

# Initialize sensors, buffers, and alert manager
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_ahtx0.AHTx0(i2c)
alert_manager = AlertManager()
temperature_buffer = DataBuffer()
humidity_buffer = DataBuffer()
sensor_updater = SensorUpdater(sensor, 2, alert_manager=alert_manager)



# Example of starting the sensor updater in a thread
def start_monitoring():
    sensor_thread = threading.Thread(target=sensor_updater.update_sensor_values, daemon=True)
    rgb_thread = threading.Thread(target=rgb_camera_thread, daemon=True)  # Thread for the RGB camera

    sensor_thread.start()
    rgb_thread.start()

def stop_monitoring():
    sensor_updater.stop()
    stop_signal.set()  # Signals the RGB thread to stop
    # Wait for threads to finish if necessary

app = Flask(__name__)
CORS(app)

@app.route('/api/graphs/room')
def get_sensor_data():
    # Fetch the latest data from the buffers
    temp_data = temperature_buffer.get_data()
    humidity_data = humidity_buffer.get_data()

    data = {
        "temperature": temp_data,
        "humidity": humidity_data,
    }

    return jsonify(data)

@app.route('/api/alerts/recent', methods=['GET'])
def get_recent_alerts():
    # Retrieve recent alerts from AlertManager
    alerts = alert_manager.get_alerts()
    # Convert alerts to a serializable format
    return jsonify(alerts)

if __name__ == "__main__":
    start_monitoring()
    app.run(host='10.0.0.209', port=5001, debug=True)
