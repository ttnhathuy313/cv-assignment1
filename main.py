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

def getAdaptiveThreshold(img, roi, num_lines = 20):
    for thresh in range(160, 230, 5):
        selected = Filter(cv2.inRange(img, thresh, 255))
        selected = cv2.bitwise_and(selected, roi)
        canny = cv2.Canny(selected, 150, 220)
        lines = cv2.HoughLinesP(canny, 1, 
                                np.pi / 180, 20, minLineLength = 5, 
                                maxLineGap = 5)
        
        # if (thresh == 200):
        #     hough = np.zeros((selected.shape[0], selected.shape[1], 3), dtype = np.uint8)
        #     print(len(lines))
        #     draw_lines(hough, lines)
        #     plt.imshow(hough)
        #     plt.show()
        if len(lines) <= num_lines:
            return thresh
    return 190


def process(img, width = 3):
    global global_threshold
    
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
    thresh = getAdaptiveThreshold(img_gray, roi)
    print(thresh)
    img_threshold = Filter(cv2.inRange(img_gray, thresh, 255))
    img_threshold_isolated = cv2.bitwise_and(img_threshold, roi)
    img_blur = img_threshold_isolated
    img_edges = cv2.Canny(img_blur, 150, 220)

    rho = 1
    theta = np.pi / 180
    threshold = 20
    min_line_len = 5
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
    for i in range(mxHeight >> 1, mxHeight - 20, 5):
        for j in range(0, mxWidth >> 1, 1):
            if (hough[i, j, 0] == 255):
                x1.append(i)
                y1.append(j)
                break
    diff = 40
    lst = -1
    for i in range(len(y1)):
        if (i > 0 and lst - y1[i] > diff):
            y1[i] = y1[i - 1]
        else:
            if (i > 0 and i < 30 and y1[i] - lst > diff):
                for j in range(0, i):
                    y1[j] = y1[i]
                continue
            lst = y1[i]

    # print(x1, y1)

    try:
        left_curve  = np.poly1d(np.polyfit(x1,y1, 1))
    except: return img
    res = img.copy()
    draw_parabol(res, left_curve)
    
    x1 = []
    y1 = []
    for i in range(mxHeight >> 1, mxHeight - 20, 5):
        found = False
        for j in range(mxWidth - 1, mxWidth >> 1, -1):
            if (hough[i, j, 0] == 255):
                x1.append(i)
                y1.append(j)
                break
    diff = 40
    lst = -1
    for i in range(len(y1)):
        if (i > 0 and y1[i] - lst > diff):
            y1[i] = y1[i - 1]
        else:
            if (i > 0 and i < 30 and lst - y1[i] > diff):
                for j in range(0, i):
                    y1[j] = y1[i]
                continue
            lst = y1[i]
    
    # plt.imshow(hough)
    # plt.show()
    try:
        right_curve = np.poly1d(np.polyfit(x1,y1, 1))
    except: return img
    
    draw_parabol(res, right_curve)
    middle_curve = (right_curve + left_curve) / 2
    draw_parabol(res, middle_curve, [255, 0, 0])
    
    
    horizontal = 300
    
    cv2.line(res, (int(left_curve(horizontal)), horizontal), 
             (int(right_curve(horizontal)), horizontal), (0, 0, 255), 2)

    dist = ((right_curve + left_curve) / 2)(horizontal) - (mxWidth >> 1)
    s = 'right ' if dist < 0 else 'left '
    distance_to_bottom_line = 7.9
    dist = round(dist * coeff * distance_to_bottom_line, 3)
    dist = abs(dist)
    lane_width = (right_curve(horizontal) - left_curve(horizontal)) * coeff * distance_to_bottom_line
    lane_width = round(lane_width, 3)
    cv2.putText(res, s + str(dist) + 'cm', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(res, 'lane width ' + str(lane_width) + 'cm', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
    #draw lane area
    points = [
        (int(left_curve(mxHeight)), mxHeight),
        (int(left_curve(mxHeight>>1)), mxHeight>>1),
        (int(right_curve(mxHeight>>1)), mxHeight>>1),
        (int(right_curve(mxHeight)), mxHeight)
    ]
    mask = np.full((mxHeight, mxWidth, 3), (255, 255, 255), dtype = np.uint8)
    
    cv2.fillPoly(mask, np.array([points]), (0, 204, 255))
    res = cv2.bitwise_and(mask, res)
    
    return res 


import yaml
from yaml.loader import SafeLoader
with open('cam.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

coeff = 3.858038022 / (data['camera_matrix'][0][0] / 3.858038022535227)


cap = cv2.VideoCapture('./sample/sample6.mp4')
print('connected')
print(cap)
while (True):
    ret, img = cap.read()
    if (ret == False):
        break
    cv2.imshow("img", process(img))
    cv2.waitKey(1)
    
# cap = cv2.VideoCapture('./sample/sample6.mp4')
# cap.set(cv2.CAP_PROP_POS_MSEC,2000) 
# ret, img = cap.read()
# plt.imshow(process(img))
# plt.show()