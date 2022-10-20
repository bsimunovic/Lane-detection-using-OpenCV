import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.core.numeric import zeros_like

def SelectHLSWhiteAndYellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    lower = np.uint8([  0, 185,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    
    lower = np.uint8([ 10,  0, 60])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined = cv2.bitwise_and(image, image, mask = mask)
   
    return combined

def HLStoBinary(image):
    hls_image = SelectHLSWhiteAndYellow(image)
    l_channel = hls_image[:,:,1]
    thresh = [0,255]
    binary = np.zeros_like(l_channel)
    binary[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 255
    return binary


def BirdEyeViewTransformation(image, image_size, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

    return warped

def BirdEyeViewTransformationInversion(image, image_size, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    unwarped = cv2.warpPerspective(image, M, image_size, flags=cv2.WARP_INVERSE_MAP)

    return unwarped

def Canny(image):
    
    binary_image = HLStoBinary(image)
    blur = cv2.GaussianBlur(binary_image, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny

def RegionOfInterest(image):
    height = image.shape[0]
    polygons = np.array([
       [[50,height], [1130,height], [900,0], [200,0]]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons, (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def isInvalid(x,y):
    max_Pixels = 3840
    invalidNumber = False
    check_x = False
    check_y = False
    if (x > max_Pixels) or (x <= 0) or x == None:
        check_x = True
    
    if (y > max_Pixels) or (y <= 0) or y == None:
        check_y = True
    
    
    invalidNumber = check_x  or check_y
    
    return invalidNumber

def DisplayLines(image, lines, oldx1,oldx2,oldy1,oldy2):
    
    lines_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            if isInvalid(x1,y1):
                x1 = int(oldx1)
                y1 = int(oldy1)
                    
            if isInvalid(x2,y2):
                x2 = int(oldx2)
                y2 = int(oldy2)
                
            cv2.line(lines_image, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 20)
    return lines_image, x1,x2,y1,y2

def makeCord(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 =int(y1*(3/7))
    x1 = int((y1- intercept)/slope)
    x2 = int((y2- intercept)/slope)
    return np.array([x1,y1,x2,y2])

def averageLaneInterception(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        paramaters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = paramaters[0]
        interception = paramaters[1]
        if slope < 0:
            left_fit.append((slope, interception))
        else:
            right_fit.append((slope, interception))
    left_line = np.array([0,0,0,0])
    right_line = np.array([0,0,0,0])
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = makeCord(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = makeCord(image, right_fit_average)

    return np.array([left_line, right_line])


def main():
    cap = cv2.VideoCapture("test.mp4")
    oldx1 = 0
    oldx2 = 0
    oldy1 = 0
    oldy2 = 0
    while(cap.isOpened()):
        #start_time = time.time()
        _, frame = cap.read()

        image_size = (frame.shape[1], frame.shape[0])
        src = np.float32(
            [(480, 480), 
            (250, 720), 
            (1300, 720),
            (900, 480)])
        dst = np.float32(
            [[(0, 0),
            (0, image_size[1]),
            (image_size[0], image_size[1]),
            (image_size[0], 0)]])
        image_copy = np.copy(frame)
        birdEyeView = BirdEyeViewTransformation(image_copy, image_size, src, dst)
        cropped_Image = RegionOfInterest(birdEyeView)
        canny_image = Canny(cropped_Image)
        

        lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 150, np.array([]), minLineLength=40, maxLineGap=10)
        if lines is None:
            lines = []
        average_line = averageLaneInterception(frame, lines)
        line_image, oldx1,oldx2,oldy1,oldy2 = DisplayLines(frame, average_line, oldx1,oldx2,oldy1,oldy2)
        birdEyeViewInversion = BirdEyeViewTransformationInversion(line_image, image_size, src, dst)

        combo_image = cv2.addWeighted(frame, 0.8, birdEyeViewInversion, 1, 1)
        

        cv2.imshow('Resultat',combo_image)
        if cv2.waitKey(1) == ord('q'):
            break
        #print("FPS: ", 1.0 / (time.time() - start_time))
    cap.relese()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

