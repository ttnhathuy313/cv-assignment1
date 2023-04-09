import cv2
import numpy as np

def draw_lines(img, lines, color = [255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                
def draw_fitted_line(img, fitted_line, color = [0, 255, 0], thickness = 2, grain=5):
    """Utility for drawing the line fitted to selected points."""
    mxHeight = img.shape[0]
    step = int((mxHeight - (mxHeight >> 1)) / grain)
    x = np.arange((mxHeight >> 1) - step, mxHeight, step)
    y = fitted_line(x)
    points = np.column_stack((y, x)).astype(np.int32)
    cv2.polylines(img, [points], False, color, thickness)
    
def Filter(img, length=10):
    """Utility for filtering out noise."""
    res = img.copy()
    res = cv2.erode(res, np.ones((length, length), np.uint8))
    res = cv2.dilate(res, np.ones((length, length), np.uint8))
    return res

def getAdaptiveThreshold(img, roi, num_lines = 20):
    """Utility for getting adaptive threshold. Choose threshold that gives
    less amount of lines after Hough Transform"""
    for thresh in range(160, 230, 5):
        selected = Filter(cv2.inRange(img, thresh, 255))
        selected = cv2.bitwise_and(selected, roi)
        canny = cv2.Canny(selected, 150, 220)
        lines = cv2.HoughLinesP(canny, 1, 
                                np.pi / 180, 20, minLineLength = 5, 
                                maxLineGap = 5)
        if lines is not None and len(lines) <= num_lines:
            return thresh
    return 190
