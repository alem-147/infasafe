import threading
import queue
import time
import cv2
import numpy as np
from jetson_inference import detectNet
import jetson_utils
from uvctypes import *

shared_regions = queue.Queue()
q = queue.Queue()

# RGB Thread functions

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

# IR Thread functions (PureThermal2 uvc examples github)

def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(
        array_pointer.contents, dtype=np.dtype(np.uint16)
    ).reshape(
        frame.contents.height, frame.contents.width
    )

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
    val = ktoc(val_k)
    cv2.putText(img, "{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

# Function to transform coordinates from RGB to IR image space
def transform_coordinates(region):
    # Constants
    RGB_WIDTH, RGB_HEIGHT = 1280, 720
    IR_WIDTH, IR_HEIGHT = 160, 120

    # Unpack region
    x1, y1, x2, y2 = region

    # Scale the coordinates
    x1_ir = int(x1 * IR_WIDTH / RGB_WIDTH)
    y1_ir = int(y1 * IR_HEIGHT / RGB_HEIGHT)
    x2_ir = int(x2 * IR_WIDTH / RGB_WIDTH)
    y2_ir = int(y2 * IR_HEIGHT / RGB_HEIGHT)

    return (x1_ir, y1_ir, x2_ir, y2_ir)

def object_detection_thread():
    # Load the object detection network
    net = detectNet('ssd-mobilenet-v2', threshold=0.5)

    # Initialize the camera or video input source
    input = jetson_utils.videoSource("/dev/video0")
    output = jetson_utils.videoOutput("display://0")

    # Create a detection region for leaves
    leaves_regions = []

    # Define the threshold for potted plant area overlapping with vase region relative to the total vase area
    area_threshold = 0.3

    while True:
        # Wait for a bit to limit frame rate to 9 Hz
        time.sleep(1/9)

        # Capture the next image
        RGB_frame = input.Capture()  # Renamed to RGB_frame

        if RGB_frame is None:  # Timeout
            continue

        # Perform object detection
        detections = net.Detect(RGB_frame, overlay='none')

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
            jetson_utils.cudaDrawRect(RGB_frame, region, (255, 127, 0, 200))
            region_ir = transform_coordinates(region)  # Transform coordinates
            shared_regions.put(region_ir)  # Add the detected leaf regions to the shared queue
            print(f"Added region {region_ir} to shared queue.")

        # Display the frame with overlays
        output.Render(RGB_frame)

        # Update the output window
        output.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        # Check for exit condition
        if not output.IsStreaming():
            shared_regions.put(None)  # Add sentinel value to indicate end of stream
            break

# The IR camera thread where we will get leaf regions from the shared queue
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

            print("Device opened!")

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                print("Device does not support Y16")
                exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                                                  frame_formats[0].wWidth, frame_formats[0].wHeight,
                                                  int(1e7 / frame_formats[0].dwDefaultFrameInterval))

            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
                print("uvc_start_streaming failed: {0}".format(res))
                exit(1)

            print("Streaming delayed to match rgb")
            time.sleep(2)

            print("Streaming started!")

            while True:

                time.sleep(1/9) # Wait for a bit to limit frame rate to 9 Hz
                
                IR_frame = q.get(True, 500)  # Retrieve the IR frame from the queue
                if IR_frame is None:
                    break
                IR_frame = cv2.resize(IR_frame[:, :], (640, 480))
                minVal, maxVal, minLoc , maxLoc = cv2.minMaxLoc(IR_frame)
                average_temperature = ktoc(np.mean(IR_frame))
                print("Average Temperature: {:.2f} °C".format(average_temperature))
                img = raw_to_8bit(IR_frame)
                display_temperature(img, minVal, minLoc, (255, 0, 0))
                display_temperature(img, maxVal, maxLoc, (0, 0, 255))

                # Get next leaf region from the queue
                if not shared_regions.empty():
                    leaf_region = shared_regions.get()
                    
                    # Check for sentinel value indicating end of stream
                    if leaf_region is None:
                        print("RGB thread has stopped. Stopping IR thread.")
                        break

                    print(f"Got leaf region {leaf_region} from shared queue.")

                    # Get subarray corresponding to the leaf_region
                    x1, y1, x2, y2 = leaf_region
                    leaf_data = q.get()  # Retrieve the IR frame from the queue
                    leaf_data = leaf_data[y1:y2, x1:x2]

                    # Calculate and print average temperature for this leaf region
                    average_temperature = ktoc(np.mean(leaf_data))
                    print(f"Average Temperature for leaf region {leaf_region}: {average_temperature:.2f} °C")


                cv2.imshow('Lepton Radiometry', img)
                cv2.waitKey(1)

            cv2.destroyAllWindows()

        finally:
                libuvc.uvc_stop_streaming(devh)
                print("Done")
                libuvc.uvc_unref_device(dev)
    finally:
        libuvc.uvc_exit(ctx)

object_detection_thread = threading.Thread(target=object_detection_thread, daemon=True)
ir_camera_thread = threading.Thread(target=ir_camera_thread, daemon=True)

object_detection_thread.start()
ir_camera_thread.start()

object_detection_thread.join()
ir_camera_thread.join()
