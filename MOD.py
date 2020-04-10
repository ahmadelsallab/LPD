# Moving Object Detection (MOD) module
import cv2
import imutils

def detect_moving_objects(frame, background):
    MIN_AREA = 10000
    cars = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # compute the absolute difference between the current frame and
    # first frame
    frame_delta = cv2.absdiff(background, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for k,c in enumerate(cnts):
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < MIN_AREA:
            continue
        car = cv2.boundingRect(c)
        cars.append(car)
    return cars