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
    
def Filter(img, length=10):
    res = img.copy()
    res = cv2.erode(res, np.ones((length, length), np.uint8))
    res = cv2.dilate(res, np.ones((length, length), np.uint8))
    return res


def process(img, width = 3):
    img = cv2.resize(img, (640, 400))
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
    img_threshold = Filter(cv2.inRange(img_gray, 160, 255))
    img_threshold_isolated = cv2.bitwise_and(img_threshold, roi)
    # plt.imshow(img_threshold_isolated, cmap = 'gray')
    # plt.show()
    img_blur = img_threshold_isolated
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
    cv2.imshow("img2", hough)
    cv2.waitKey(1)

    # plt.imshow(hough, cmap = 'gray')
    # plt.show()

    x1 = []
    y1 = []
    minY = mxWidth
    id = -1
    allowed_width = 15
    for i in range (mxHeight >> 1, mxHeight - 20, 5):
        for j in range(0, mxWidth >> 1):
            if hough[i, j, 0] == 255:
                if (j < minY):
                    minY = j
                    id = i
                break
        if (minY <= 10):
            break

    lst = minY
    wo = 0
    for i in range(id, mxHeight - 20, 2):
        found = False
        for j in range(0, mxWidth >> 1):
            if (hough[i, j, 0] == 255 and abs(j - lst) < allowed_width):
                x1.append(i)
                y1.append(j)
                # hough = cv2.circle(hough, (j, i), 2, (0, 0, 255), 2)
                lst = j
                found = True
                break
        if (not found):
            wo += 1
        else:
            wo = 0
        if (wo> 5):
            break
    lst = minY
    wo = 0
    for i in range(id, mxHeight >> 1, -5):
        found = False
        for j in range(0, mxWidth >> 1):
            if (hough[i, j, 0] == 255 and abs(j - lst) < allowed_width):
                x1.append(i)
                y1.append(j)
                # hough = cv2.circle(hough, (j, i), 2, (0, 0, 255), 2)
                lst = j
                found = True
                break

    # print(x1, y1)

    try:
        left_curve  = np.poly1d(np.polyfit(x1,y1, 2))
    except: return img
    res = img.copy()
    draw_parabol(res, left_curve)
    
    maxY = 0
    id = -1
    allowed_width = 15
    for i in range (mxHeight >> 1, mxHeight - 20, 5):
        for j in range(mxWidth - 1, mxWidth >> 1, -1):
            if hough[i, j, 0] == 255:
                if (j > maxY):
                    maxY = j
                    id = i
                break
        if (maxY >= mxWidth - 10):
            break
    
    x1 = []
    y1 = []
    
    lst = maxY
    wo = 0
    for i in range(id, mxHeight - 20, 2):
        found = False
        for j in range(mxWidth - 1, mxWidth >> 1, -1):
            if (hough[i, j, 0] == 255 and abs(j - lst) < allowed_width):
                x1.append(i)
                y1.append(j)
                # hough = cv2.circle(hough, (j, i), 2, (0, 0, 255), 2)
                lst = j
                found = True
                break
        if (not found):
            wo += 1
        else:
            wo = 0
        if (wo> 5):
            break
    lst = maxY
    wo = 0
    for i in range(id, mxHeight >> 1, -2):
        found = False
        for j in range(mxWidth - 1, mxWidth >> 1, -1):
            if (j > mxWidth): break
            if (hough[i, j, 0] == 255 and abs(j - lst) < allowed_width):
                x1.append(i)
                y1.append(j)
                # hough = cv2.circle(hough, (j, i), 2, (0, 0, 255), 2)
                lst = j
                found = True
                break

    # plt.imshow(hough)
    # plt.show()
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


import yaml
from yaml.loader import SafeLoader
with open('cam.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

coeff = 3.858038022 / (data['camera_matrix'][0][0] / 3.858038022535227)

cap = cv2.VideoCapture('./sample/sample3.mp4')
print('connected')
print(cap)
while (True):
    ret, img = cap.read()
    if (ret == False):
        break
    cv2.imshow("img", process(img))
    cv2.waitKey(1)
    
# cap = cv2.VideoCapture('./sample/sample3.mp4')
# cap.set(cv2.CAP_PROP_POS_MSEC,6000) 
# ret, img = cap.read()
# plt.imshow(process(img))
# plt.show()