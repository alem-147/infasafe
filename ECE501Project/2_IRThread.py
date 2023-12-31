import threading
import cv2
import numpy as np
from uvctypes import *
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

BUF_SIZE = 2
q = Queue(BUF_SIZE)

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
    val = ktof(val_k)
    cv2.putText(img, "{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
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

            try:
                while True:
                    data = q.get(True, 500)
                    if data is None:
                        break
                    data = cv2.resize(data[:, :], (640, 480))
                    minVal, maxVal, _, _ = cv2.minMaxLoc(data)
                    average_temperature = ktoc(np.mean(data))
                    print("Average Temperature: {:.2f} °C".format(average_temperature))
                    img = raw_to_8bit(data)
                    display_temperature(img, minVal, (0, 0), (255, 0, 0))
                    display_temperature(img, maxVal, (img.shape[1] - 200, 0), (0, 0, 255))
                    cv2.imshow('Lepton Radiometry', img)
                    cv2.waitKey(1)

                cv2.destroyAllWindows()
            finally:
                libuvc.uvc_stop_streaming(devh)

            print("Done")
        finally:
            libuvc.uvc_unref_device(dev)
    finally:
        libuvc.uvc_exit(ctx)

# Create a thread for the IR camera
ir_camera_thread = threading.Thread(target=ir_camera_thread)

# Start the thread
ir_camera_thread.start()

# Wait for the thread to finish
ir_camera_thread.join()

