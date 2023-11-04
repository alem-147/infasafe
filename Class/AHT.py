import threading
import time
import board
import adafruit_ahtx0

class SensorUpdater(threading.Thread):
    def __init__(self):
        super(SensorUpdater, self).__init__()
        self.sensor = adafruit_ahtx0.AHTx0(board.I2C())  # Initialize the sensor
        self.current_temperature = 0.0
        self.current_humidity = 0.0
        self.running = True

    def run(self):
        while self.running:
            self.current_temperature = self.sensor.temperature
            self.current_humidity = self.sensor.relative_humidity
            time.sleep(2)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    # ... Initialization code ...

    # Create an instance of the SensorUpdater class
    sensor_updater = SensorUpdater()

    # Start the sensor updater thread
    sensor_updater.start()

    try:
        while True:
            # Access current temperature and humidity values from the SensorUpdater instance
            temperature = sensor_updater.current_temperature
            humidity = sensor_updater.current_humidity

            # Use the temperature and humidity values as needed
            print(f"Temperature: {temperature:.1f}°C, Humidity: {humidity:.1f}%")

            time.sleep(1)  # Adjust the sleep interval as needed

    except KeyboardInterrupt:
        # Stop the sensor updater thread when Ctrl+C is pressed
        sensor_updater.stop()
        sensor_updater.join()

    # ... Rest of your code ...
