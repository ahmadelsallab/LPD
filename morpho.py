import cv2
import numpy as np
from utils import *

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 40

def extractValue(img):
    height, width, numChannels = img.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, img = cv2.split(img)

    return img
def maximizeContrast(gray):

    height, width = gray.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# Denosing + Gray + Thresholding
def preprocess(img):
    
    gray = extractValue(img)

    gray = maximizeContrast(gray)
    

    height, width = gray.shape

    blurred = np.zeros((height, width, 1), np.uint8)

    blurred = cv2.GaussianBlur(gray, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    #thresh = cv2.adaptiveThreshold(blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    thresh = cv2.adaptiveThreshold(blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return thresh



def detect_LP_morpho(img, L_min=0, L_max=1000, W_min=0, W_max=1000, debug=False):

    min_canny = 100
    max_canny = 200
    dilation_type = cv2.MORPH_RECT #cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS

    
    thresh = preprocess(img)
    if debug: plot_img(thresh)
    
    edges = cv2.Canny(thresh,100,200)
    if debug: plot_img(edges)
    
    kernel_sz = 3
    iterations = 2
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sz, kernel_sz))
    dilated = cv2.dilate(edges, structuringElement, iterations=iterations)
    
    
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    candidates = []

    for c in contours:
        rotrect = cv2.minAreaRect(c)
        center, size, theta = rotrect
        L, W = min(size), max(size)
  
            
        if L >= L_min and L <= L_max and W >= W_min and W <= W_max:
            #rec = rotrect
            candidates.append(rotrect)
            box = cv2.boxPoints(rotrect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, SCALAR_GREEN, 2)
            
            if debug:
                text = 'L=' + str(int(L)) + ', W=' + str(int(W))
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = float(L) / 40.0                    # base font scale on height of plate area
                thickness = int(round(scale * 1.5))
                color = SCALAR_YELLOW
                cv2.putText(img, text, (int(center[0]-L/2), int(center[1]-W/2)), font, scale, color, thickness)
                #cv2.putText(img, text,  (0,10), font, int(scale), color, int(thickness))
    
    if debug: plot_img(dilated)
        
    #print(rec)
    return img, candidates

def detect_LP(img, debug=False):
    sz = (img.shape[1], img.shape[0]) 
    car_LP, LPs = detect_LP_morpho(cv2.resize(img, (500,500)), L_min=35, L_max=60, W_min=55, W_max=120, debug=debug)
    car_LP = cv2.resize(car_LP, sz)
    return car_LP, LPs
