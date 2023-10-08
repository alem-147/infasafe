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


# Future TODO
# event log for when nose is not detected - consider more events
#   tuple of timestamp and status


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

input = videoSource("csi://0")
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

        # TODO - set reigons based on links
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
            # dealing with reversable values -> posenet doesn't deal with upsideown
            # left_bound, right_bound = (leye.x, reye.x) if leye.x < reye.x else (reye.x, leye.x)

            # can always assume posenet will have left eye to the left of right
            # this is presumeably due to training data
            left_bound, right_bound = leye.x, reye.x
            upper_bound, lower_bound = (min(leye.y,reye.y),nose.y)

            # breathROI=((left_bound, nose.y, right_bound,nose.y+50))
        elif reyeidx > 0:
            nose = pose.Keypoints[noseidx]
            reye = pose.Keypoints[reyeidx]
            # left_bound, right_bound = (nose.x, reye.x) if nose.x < reye.x else (reye.x, nose.x)
            # upper_bound, lower_bound = (min(reye.y, nose.y),max(reye.y,nose.y))

            left_bound, right_bound = nose.x, reye.x
            upper_bound, lower_bound = reye.y, nose.y
        elif leyeidx > 0:
            nose = pose.Keypoints[noseidx]
            leye = pose.Keypoints[leyeidx]
            # left_bound, right_bound = (nose.x, leye.x) if nose.x < leye.x else (leye.x, nose.x)
            # upper_bound, lower_bound = (min(leye.y, nose.y),max(leye.y,nose.y))

            left_bound, right_bound = leye.x, nose.x
            upper_bound, lower_bound = leye.y, nose.y
        # neither is seen
        else:
            left_bound, right_bound = (nose.x-50, nose.x+50)
            upper_bound, lower_bound = (nose.y-50,nose.y+50)          
        thermalROI = (left_bound,upper_bound,right_bound,lower_bound)
        breathROI = (left_bound,nose.y,right_bound,nose.y+(nose.y-upper_bound)*.75)
        cudaDrawRect(img,thermalROI,(255,255,255,200))
        cudaDrawRect(img,breathROI,(0,255,255,200))

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