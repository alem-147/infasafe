#!/usr/bin/env python3

try:
    import cv2
except ImportError:
    print("ERROR: python-opencv must be installed")
    exit(1)

class OpenCvCapture(object):

    def __init__(self):
        for i in reversed(range(10)):
            print("Testing for presence of camera")
            cv2_cap = cv2.VideoCapture(i)
            if cv2_cap.isOpened():
                break

        if not cv2_cap.isOpened():
            print("Camera not found!")
            exit(1)

        self.cv2_cap = cv2_cap

    def show_video(self):
        cv2.namedWindow("lepton", cv2.WINDOW_NORMAL)
        print("Running, ESC or Ctrl-c to exit...")
        while True:
            ret, img = self.cv2_cap.read()

            if not ret:
                print("Error reading image")
                break

            cv2.imshow("lepton", cv2.resize(img, (640, 480)))
            if cv2.waitKey(5) == 27:
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    OpenCvCapture().show_video()