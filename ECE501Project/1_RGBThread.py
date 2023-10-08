import threading
import cv2
import numpy as np
from jetson_inference import detectNet
import jetson_utils

def calculate_area(region):
    left, top, right, bottom = region
    return (right - left) * (bottom - top)

def calculate_intersection_area(region1, region2):
    x_left = max(region1[0], region2[0])
    y_top = max(region1[1], region2[1])
    x_right = min(region1[2], region2[2])
    y_bottom = min(region1[3], region2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area

def object_detection_thread():
    # Load the object detection network
    net = detectNet('ssd-mobilenet-v2', threshold=0.5)

    # Initialize the camera or video input source
    input = jetson_utils.videoSource("/dev/video1")
    output = jetson_utils.videoOutput("display://0")

    # Create a detection region for leaves
    leaves_regions = []

    # Define the threshold for potted plant area overlapping with vase region relative to the total vase area
    area_threshold = 0.3

    while True:
        # Capture the next image
        frame = input.Capture()

        if frame is None:  # Timeout
            continue

        # Perform object detection
        ##detections = net.Detect(frame, overlay="none")
        detections = net.Detect(frame)

        # Clear previous leaves regions
        leaves_regions = []

        # Initialize variables for vase and potted plant regions
        vase_regions = []
        potted_plant_regions = []

        # Iterate over detections
        for detection in detections:
            # Get class label and bounding box coordinates
            class_label = detection.ClassID
            left = int(detection.Left)
            top = int(detection.Top)
            right = int(detection.Right)
            bottom = int(detection.Bottom)

            # Print detection information
            print("<detectNet.Detection object>")
            print("   -- Confidence: {:.6f}".format(detection.Confidence))
            print("   -- ClassID: {}".format(class_label))
            print("   -- Left:    {}".format(left))
            print("   -- Top:     {}".format(top))
            print("   -- Right:   {}".format(right))
            print("   -- Bottom:  {}".format(bottom))
            print("   -- Width:   {}".format(right - left))
            print("   -- Height:  {}".format(bottom - top))
            print("   -- Area:    {:.1f}".format((right - left) * (bottom - top)))
            print("   -- Center:  ({}, {})".format((left + right) / 2, (top + bottom) / 2))

            # Check if the detection is a potted plant
            if class_label == 64:  # Class ID for 'potted plant'
                potted_plant_regions.append((left, top, right, bottom))

            # Check if the detection is a vase
            if class_label == 86:  # Class ID for 'vase'
                vase_regions.append((left, top, right, bottom))

        # Iterate over potted plant regions
        for potted_plant_region in potted_plant_regions:
            nearest_vase_region = None
            min_distance = float('inf')

            # Find the nearest vase region
            for vase_region in vase_regions:
                distance = np.linalg.norm(np.array(potted_plant_region[:2]) - np.array(vase_region[:2]))
                if distance < min_distance:
                    min_distance = distance
                    nearest_vase_region = vase_region

            # Check if a nearest vase region is found
            if nearest_vase_region is not None:
                # Calculate the intersection area between the potted plant and vase regions
                intersection_area = calculate_intersection_area(potted_plant_region, nearest_vase_region)

                # Calculate the total vase area
                total_vase_area = sum(calculate_area(region) for region in vase_regions)

                # Check if the intersection area exceeds the threshold relative to the total vase area
                if intersection_area >= area_threshold * total_vase_area:
                    # Remove the vase region from the potted plant region to create the leaf region
                    leaf_region = (
                        potted_plant_region[0],
                        potted_plant_region[1],
                        potted_plant_region[2],
                        min(potted_plant_region[3], nearest_vase_region[1])
                    )
                    leaves_regions.append(leaf_region)

        # Draw rectangles on the frame for leaf regions
        for region in leaves_regions:
            jetson_utils.cudaDrawRect(frame, region, (255, 127, 0, 200))

        # Display the frame with overlays
        output.Render(frame)

        # Update the output window
        output.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        # Check for exit condition
        if not output.IsStreaming():
            break

# Create a thread for object detection
object_detection_thread = threading.Thread(target=object_detection_thread)

# Start the thread
object_detection_thread.start()

# Wait for the thread to finish
object_detection_thread.join()

