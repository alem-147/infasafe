keypointsDic = {0 : "nose",
                1 : "left_eye",
                2 : "right_eye",
                3 : "left_ear",
                4 : "right_ear",
                5 : "left_shoulder",
                6 : "right_shoulder",
                7 : "left_elbow",
                8 : "right_elbow",
                9 : "left_wrist",
                10 : "right_wrist",
                11 : "left_hip",
                12 : "right_hip",
                13 : "left_knee",
                14 : "right_knee",
                15 : "left_ankle",
                16 : "right_ankle",
                17 : "neck"}

def poseParts(poses: list):
    parts = []
    for pose in poses:
        for keypoints in pose.Keypoints:
            id = keypoints.ID
            parts.append(keypointsDic[id])
    return parts

def multipleObjects(poses :list):
    status = True
    if len(poses) != 1:
        status = False
    return status

def updateArray(EstimationArray : list, status :bool):
    EstimationArray.insert(0, status)
    EstimationArray.pop()
    
def poseEstimatorNose(PoseEstimationArray : list, poses :list):
    parts = poseParts(poses)
    
    if "neck" in parts:
        updateArray(PoseEstimationArray, True)
        return
    updateArray(PoseEstimationArray, False)
        
def neckEstimatorEye(NeckEstimationArray : list, poses):
    parts = poseParts(poses)
    
    if "left_eye" and "right_eye" in parts:
        updateArray(NeckEstimationArray, True)
        return
    updateArray(NeckEstimationArray, False)
    
def neckEstimatorNeck(NeckEstimationArray : list, poses):
    parts = poseParts(poses)
    
    if "neck" in parts:
        updateArray(NeckEstimationArray, True)
        return
    updateArray(NeckEstimationArray, False)

def thermalEstimatorFace(ThermalEstimationArray : list, poses, tempQueue, img, font):
    if not tempQueue.empty(): 
            t_img = tempQueue.get()
    
    t_img = round((t_img * 9/5) + 32, 1)
    
    if (poses is not None) and (t_img is not None):
        for nose in poses:
            box = np.array([int(nose.Left), int(nose.Top), int(nose.Right), int(nose.Bottom)])/40
            (startX, startY, endX, endY) = box.astype("int")
            if startX>0 : startX -= 1
            if startY>0 : startY -= 1          
            if endX <32 : endX += 1
            if endY <18 : endY += 1
            tmax = t_img[startY:endY, startX:endX].max()
            text = "Tmax={:.1f} C".format(tmax)
            font.OverlayText(img, img.width, img.height, text, int(nose.Left), int(nose.Top), font.White, font.Gray40)
    
    updateArray(ThermalEstimationArray, t_img)

def warning(EstimationArray : list, wantedPercent : int):
    avg = sum(1 for status in EstimationArray if status == True) / len(EstimationArray)
    avg = round(avg *100, 2)
    if avg < wantedPercent:
        return False, avg
    return True, avg

def warningThermal(ThermalEstimationArray : list, upper, lower):
    avg = sum(ThermalEstimationArray) /len(ThermalEstimationArray)
    if avg < upper and avg > lower:
        return True, avg
    return False, avg

def tempToBool(temp, upper, lower):
    if temp < upper and temp > lower:
        return True
    return False