#!/usr/bin/env python3

import time
import sys
import argparse
import thresholds as t
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaFont

from multiprocessing import Process
from multiprocessing import Queue

#set threshold and define model type for poseNet
normthreshold = 0.15
multthreshold = 0.10

# load the pose estimation model
net = poseNet("resnet18-body", threshold = normthreshold )
multnet = poseNet("resnet18-body", threshold = multthreshold )

# create video sources & outputs
input = videoSource("/dev/video0")
output = videoOutput("display://0")

#load font
font = cudaFont()

problemNames = ["Mult", "Eye", "Neck", "Nose", "Thermal"]
fieldArray = ["TempActual", "TempStatus", "NeckEyeStatus", "NeckNeckStatus", "PoseNoseStatus", "MultipleStatus", "RoomStatus"]
timestamp = []

arraySize = 600
upperTemp = 99.5
lowerTemp = 97.5
#each test needs its own list
MultipleEstimation = [True]*arraySize
NeckAngleEstimationEye = [True]*arraySize
NeckAngleEstimationNeck = [True]*arraySize
PoseEstimationNose = [True]*arraySize
ThermalEstimation = [True]*arraySize


# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # perform pose estimation (with overlay)
    
    poses = net.Process(img, overlay="links,keypoints")
    multposes = multnet.Process(img, overlay = "links, keypoints")

    
    
    mult = t.multipleObjects(multposes)
    t.updateArray(MultipleEstimation, mult)
    
    if mult:
        t.neckEstimatorEye(NeckAngleEstimationEye, poses)
        t.neckEstimatorNeck(NeckAngleEstimationNeck, poses)
        t.poseEstimatorNose(PoseEstimationNose, poses)
        
    else:
        t.updateArray(NeckAngleEstimationEye, False)
        t.updateArray(NeckAngleEstimationNeck, False)
        t.updateArray(PoseEstimationNose, False)

    # render the image
    output.Render(img)

    #will need these for thingSpeak
    #mult, eyes, neck, nose, thermal
    statusMult, averageMult = t.warning(MultipleEstimation, 20)
    statusEye, averageEye = t.warning(NeckAngleEstimationEye, 50)
    statusNeck, averageNeck = t.warning(NeckAngleEstimationNeck, 80)
    statusNose, averageNose = t.warning(PoseEstimationNose, 50)
    statusTherm, averageTherm = t.warningThermal(ThermalEstimation, upperTemp, lowerTemp)
    
    statuses = [statusMult, statusEye, statusNeck, statusNose, statusTherm]
    averages = [averageMult, averageEye, averageNeck, averageNose, averageTherm]
    statusList = list(zip(problemNames, statuses, averages))
    
    #fieldArray = ["TempActual", "TempStatus", "NeckEyeStatus", "NeckNeckStatus", "PoseNoseStatus", "MultipleStatus", "RoomStatus"]
    intValues = []
    temp = ThermalEstimation[0]
    tempBool = t.tempToBool(temp, upperTemp, lowerTemp)
    neckEyeBool = int(NeckAngleEstimationEye[0])
    neckNeckBool = int(NeckAngleEstimationNeck[0])
    poseNoseBool = int(PoseEstimationNose[0])
    mult = int(mult)
    
    dataArray = [temp, tempBool, neckEyeBool, neckNeckBool, poseNoseBool, mult ]
    
    for triples in statusList:
        print(triples[0], "- Status: ", triples[1], ", Average: " , triples[2])
    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
