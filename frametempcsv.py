import sys
import time
import os
import csv
from datetime import datetime
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, saveImageRGBA, cudaDeviceSynchronize
import board
import adafruit_ahtx0


net = poseNet('resnet18-body', threshold=0.15)

# Initialize the camera or video input source
input = videoSource("/dev/video0")
output = videoOutput("display://0")

# Create folder for RGB frames
if not os.path.exists("rgbframes"):
    os.makedirs("rgbframes")

# Create CSV file for temperature and humidity data
csv_filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Temperature (C)', 'Humidity (%)'])

# Create sensor object for temperature and humidity sensor
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_ahtx0.AHTx0(i2c)

while True:
    # Capture the next image
    img = input.Capture()

    if img is None:  # Timeout
        continue  

    # Perform pose estimation (with overlay)
    poses = net.Process(img)

    # Print the pose results
    print("Detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)

    # Save the RGB frame
    frame_filename = f"rgbframes/frame_{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}.jpg"
    cudaDeviceSynchronize()
    saveImageRGBA(frame_filename, img, 1280, 720)
    
    # Get temperature and humidity data
    temperature = sensor.temperature
    humidity = sensor.relative_humidity

    # Write temperature and humidity data to the CSV file
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    csv_writer.writerow([timestamp, temperature, humidity])
    csv_file.flush()

    # Render the image
    output.Render(img)

    # Update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format("resnet18-body", net.GetNetworkFPS()))

    # Print out performance info
    net.PrintProfilerTimes()

    # Exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

# Close the CSV file
csv_file.close()
