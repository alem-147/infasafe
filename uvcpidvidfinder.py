from uvctypes import *
import time
import cv2
import numpy as np
try:
  from queue import Queue
except ImportError:
  from Queue import Queue
import platform

def print_device_info(device_handle):
    """
    Print the device's information, including VID and PID.
    """
    dev_desc = uvc_device_descriptor()
    if libuvc.uvc_get_device_descriptor(device_handle, byref(dev_desc)) == 0:
        print("Device found: VID={:04x}, PID={:04x}".format(dev_desc.idVendor, dev_desc.idProduct))
        libuvc.uvc_free_device_descriptor(byref(dev_desc))
    else:
        print("Could not retrieve device descriptor")

def main():
    ctx = POINTER(uvc_context)()
    dev_list = POINTER(POINTER(uvc_device))()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(1)

    # Attempt to list all UVC devices
    res = libuvc.uvc_get_device_list(ctx, byref(dev_list))
    if res < 0:
        print("uvc_get_device_list error")
        exit(1)

    # Iterate through the list of devices and print their information
    i = 0
    while dev_list[i]:
        dev = dev_list[i]
        i += 1
        res = libuvc.uvc_open(dev, byref(devh))
        if res < 0:
            print("Could not open device: ", res)
        else:
            print_device_info(devh)
            libuvc.uvc_close(devh)
    
    libuvc.uvc_free_device_list(dev_list, 1)
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
    main()

