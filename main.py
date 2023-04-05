import cv2
import numpy as np 
import matplotlib.pyplot as plt

def draw_lines(img, lines, color = [255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                
def draw_parabol(img, parabol, color = [0, 255, 0], thickness = 2, grain=5):
    """Utility for drawing parabol."""
    mxHeight = img.shape[0]
    step = int((mxHeight - (mxHeight >> 1)) / grain)
    x = np.arange((mxHeight >> 1) - step, mxHeight, step)
    y = parabol(x)
    points = np.column_stack((y, x)).astype(np.int32)
    cv2.polylines(img, [points], False, color, thickness)


def process(img, width = 3):
    mxHeight = img.shape[0]
    mxWidth = img.shape[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    roi_vertices = np.array([[
        [0, mxHeight - 1],
        [0, mxHeight >> 1],
        [mxWidth - 1, mxHeight >> 1],
        [mxWidth - 1, mxHeight - 1]    
    ]], dtype = np.int32)

    ignore_mask_color = 255
    if (len(img_gray.shape) > 2):
        num_channel = img_gray.shape[2]
        ignore_mask_color = (255, ) * num_channel
    roi_mask = np.zeros_like(img_gray)
    roi = cv2.fillPoly(roi_mask, roi_vertices, ignore_mask_color)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_threshold = cv2.inRange(img_gray, 180, 255)
    img_threshold_isolated = cv2.bitwise_and(img_threshold, roi)
    img_blur = cv2.GaussianBlur(img_threshold_isolated, (11, 11), 15, 15)
    img_edges = cv2.Canny(img_blur, 150, 220)

    rho = 1
    theta = np.pi / 180
    threshold = 20
    min_line_len = 0
    max_line_gap = 5

    lines = cv2.HoughLinesP(
        img_edges, rho, theta, threshold, minLineLength = min_line_len, maxLineGap = max_line_gap)

    hough = np.zeros((img_edges.shape[0], img_edges.shape[1], 3), dtype = np.uint8)
    draw_lines(hough, lines)

    x1 = []
    y1 = []
    for i in range (mxHeight >> 1, mxHeight - 20, 10):
        if (i > mxHeight): break
        for j in range(0, mxWidth >> 1):
            if (j > mxWidth): break
            if hough[i, j, 0] == 255:
                x1.append(i)
                y1.append(j)
                break

    try:
        left_curve  = np.poly1d(np.polyfit(x1,y1, 2))
    except: return img
    res = img.copy()
    draw_parabol(res, left_curve)

    x1 = []
    y1 = []
    for i in range (mxHeight >> 1, mxHeight - 20, 10):
        if (i > mxHeight): break
        for j in range(mxWidth - 1, mxWidth >> 1, -1):
            if (j > mxWidth): break
            if hough[i, j, 0] == 255:
                x1.append(i)
                y1.append(j)
                break

    try:
        right_curve = np.poly1d(np.polyfit(x1,y1, 2))
    except: return img
    
    draw_parabol(res, right_curve)
    draw_parabol(res, (right_curve + left_curve) / 2, [255, 0, 0])
    
    dist = ((right_curve + left_curve) / 2)(mxHeight) - (mxWidth >> 1)
    s = 'left ' if dist < 0 else 'right '
    dist = round(dist * coeff * 50, 3)
    dist = abs(dist)
    cv2.putText(res, s + str(dist) + 'mm', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    
    return res 

#400, 640

import yaml
from yaml.loader import SafeLoader
with open('cam.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

coeff = 3.858038022 / (data['camera_matrix'][0][0] / 3.858038022535227)

cap = cv2.VideoCapture(0)
print('connected')
print(cap)
while (True):
    ret, img = cap.read()
    if (ret == False):
        break
    cv2.imshow("img", process(img))
    cv2.waitKey(1)
    
cap = cv2.VideoCapture('two_lanes.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC,12000) 
ret, img = cap.read()
