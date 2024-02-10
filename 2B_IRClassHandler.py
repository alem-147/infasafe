#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from uvctypes import *
import cv2
import numpy as np
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

class IRCameraHandler:
    def __init__(self, buf_size=2, vid=0x1e4e, pid=0x0100):
        self.BUF_SIZE = buf_size
        self.VID = vid
        self.PID = pid
        self.q = Queue(self.BUF_SIZE)
        self.ctx = POINTER(uvc_context)()
        self.dev = POINTER(uvc_device)()
        self.devh = POINTER(uvc_device_handle)()
        self.ctrl = uvc_stream_ctrl()
        self.initialize_camera()

    def py_frame_callback(self, frame, _):
        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        data = np.fromiter(array_pointer.contents, dtype=np.uint16).reshape(frame.contents.height, frame.contents.width)
        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            return
        if not self.q.full():
            self.q.put(data)

    def initialize_camera(self):
        res = libuvc.uvc_init(byref(self.ctx), 0)
        if res < 0:
            print("uvc_init error")
            exit(1)

        res = libuvc.uvc_find_device(self.ctx, byref(self.dev), self.VID, self.PID, None)
        if res < 0:
            print(f"uvc_find_device error for VID: {self.VID:04x} PID: {self.PID:04x}")
            exit(1)

        res = libuvc.uvc_open(self.dev, byref(self.devh))
        if res < 0:
            print("uvc_open error")
            exit(1)

        print("Device opened!")

    def start_streaming(self):
        PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(self.py_frame_callback)

        libuvc.uvc_get_stream_ctrl_format_size(self.devh, byref(self.ctrl), UVC_FRAME_FORMAT_Y16, 160, 120, 9)
        res = libuvc.uvc_start_streaming(self.devh, byref(self.ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
        if res < 0:
            print(f"uvc_start_streaming failed: {res}")
            exit(1)

        try:
            while True:
                data = self.q.get(True, 500)
                self.process_frame(data)
        finally:
            self.stop_streaming()

    def process_frame(self, data):
        data = cv2.resize(data, (640, 480))
        img = self.raw_to_8bit(data)
        cv2.imshow('IR Image', img)
        cv2.waitKey(1)

    def stop_streaming(self):
        libuvc.uvc_stop_streaming(self.devh)
        print("Streaming stopped.")
        self.cleanup()

    def cleanup(self):
        libuvc.uvc_unref_device(self.dev)
        libuvc.uvc_exit(self.ctx)
        print("Cleaned up and exited.")

    @staticmethod
    def raw_to_8bit(data):
        cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(data, 8, data)
        return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

if __name__ == '__main__':
    ir_camera_handler = IRCameraHandler()
    ir_camera_handler.start_streaming()

