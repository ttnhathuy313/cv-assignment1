import cv2
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from utils import getAdaptiveThreshold, draw_lines, draw_fitted_line, Filter

frame_count = 0
global_threshold = 190
roi = None


def process(img):
    global frame_count
    global roi
    frame_count += 1
    
    # resize image to 640x400
    img = cv2.resize(img, (640, 400))
    mxHeight = img.shape[0]
    mxWidth = img.shape[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if (type(roi) != np.ndarray):
        roi_vertices = np.array([[
            [0, mxHeight - 1],
            [0, mxHeight >> 1],
            [mxWidth - 1, mxHeight >> 1],
            [mxWidth - 1, mxHeight - 1]    
        ]], dtype = np.int32)
        
        ignore_mask_color = 255
        roi_mask = np.zeros_like(img_gray)
        roi = cv2.fillPoly(roi_mask, roi_vertices, ignore_mask_color)
        
    thresh = global_threshold
    #re calculate threshold every 5 frames
    if (frame_count % 5 == 1): thresh = getAdaptiveThreshold(img_gray, roi)
    img_threshold = Filter(cv2.inRange(img_gray, thresh, 255))
    img_threshold_isolated = cv2.bitwise_and(img_threshold, roi)
    img_edges = cv2.Canny(img_threshold_isolated, 150, 220)

    threshold = 20
    min_line_len = 5
    max_line_gap = 5

    lines = cv2.HoughLinesP(
        img_edges, 1, np.pi / 180, threshold, 
        minLineLength = min_line_len, maxLineGap = max_line_gap)


    hough = np.zeros((img_edges.shape[0], img_edges.shape[1], 3), dtype = np.uint8)
    draw_lines(hough, lines)
    cv2.imshow("hough lines transform", hough)
    cv2.waitKey(1)
    
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

    try:
        left_curve  = np.poly1d(np.polyfit(x1,y1, 1))
    except: return img
    res = img.copy()
    draw_fitted_line(res, left_curve)
    
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
    
    #end mess
    
    try:
        right_curve = np.poly1d(np.polyfit(x1,y1, 1))
    except: return img
    
    draw_fitted_line(res, right_curve)
    middle_curve = (right_curve + left_curve) / 2
    draw_fitted_line(res, middle_curve, [255, 0, 0])
    
    
    horizontal = 300
    
    cv2.line(res, (int(left_curve(horizontal)), horizontal), 
             (int(right_curve(horizontal)), horizontal), (0, 0, 255), 2)

    dist = ((right_curve + left_curve) / 2)(horizontal) - (mxWidth >> 1)
    s = 'right ' if dist < 0 else 'left '
    distance_to_bottom_line = 7.9 #this param seems to be working idk why
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


#getting camera matrix from yaml file
import yaml
from yaml.loader import SafeLoader
with open('cam.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

#to do: find a way to get this value from the camera
coeff = 3.858038022 / (data['camera_matrix'][0][0] / 3.858038022535227)

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help="indicate path to video", 
                    default='./sample/sample5.mp4')
parser.add_argument('--real-time', type=bool, 
                    help="set this to true if you want to calculate real time", 
                    default=False)
args = parser.parse_args()

if __name__ == "__main__":
    cap = cv2.VideoCapture(args.video)
    if (args.real_time):
        cap = cv2.VideoCapture(0)
    while (True):
        ret, img = cap.read()
        if (ret == False):
            break
        cv2.imshow("img", process(img))
        cv2.waitKey(1)
# cap = cv2.VideoCapture(args.video)
# cap.set(cv2.CAP_PROP_POS_MSEC,6000) 
# ret, img = cap.read()
# plt.imshow(process(img))
# plt.show()