#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaDrawRect

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
# input = videoSource(args.input, argv=sys.argv)
# output = videoOutput(args.output, argv=sys.argv)

input = videoSource("/dev/video0")
output = videoOutput("display://0")

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('ROI', pose.ROI)
        print('Links', pose.Links)

        # TODO - draw reigon of interest about the mouth for thermal imaging
        # check for each eye, define reigon as just above each eye
        # for missing eye, define relative position of in place eye to nose
        # and go in oppiste direction half the distance for roi
        noseidx = pose.FindKeypoint('nose')
        leyeidx = pose.FindKeypoint('left_eye')
        reyeidx = pose.FindKeypoint('right_eye')
        thermalROI = (0,0,0,0)
        breathROI = (0,0,0,0)
        if noseidx < 0 :
            continue
        # both eyes
        if leyeidx > 0 and reyeidx > 0:
            reye = pose.Keypoints[reyeidx]
            leye = pose.Keypoints[leyeidx]
            nose = pose.Keypoints[noseidx]
            left_bound, right_bound = (leye.x, reye.x) if leye.x < reye.x else (reye.x, leye.x)
            # TODO dynamically define by chest existance but for now just hardcode
            thermalROI=(left_bound-50, min(leye.y,reye.y), right_bound+50,nose.y)
            breathROI=((left_bound, nose.y, right_bound,nose.y+50))
        elif reyeidx > 0:
            continue
        elif leyeidx > 0:
            continue
        # should neither is seen
        else:
            continue
        cudaDrawRect(img,thermalROI,(255,255,255,200))
        cudaDrawRect(img,breathROI,(255,255,255,200))

    # TODO - checkout conversion to numpy array for cv2 and matplotlib
    # cudaToNumpy -> https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md
    


    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break