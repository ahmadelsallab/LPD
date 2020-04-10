import argparse
from utils import *
from mod import detect_moving_objects
import numpy as np
import cv2
import imutils
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_video(video_file, # The video path to be processed
                  video_output_file, # The output video file
                  output_video_resolution=(640,480), # The desired output resolution
                  frames_cnt=-1, # The desired number of frames to process. If None the whole video is processed.
                  cars_detection=True, # LPD will work on the car cropped image or whole image.
                  show_cars_bbox=0,# 0=dont show, 1: show rect, 2: show oriented bbox.
                  detect_LP_fn=None, # The LPD function.
                  debug=False):

    	

    # Set input capture to 1080p
    cap = cv2.VideoCapture(video_file)
    make_1080p(cap)
    
    # Set the frame count if no passed desired number process the whole video
    if frames_cnt == -1: frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the output video file
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_output_file,fourcc, 20.0, output_video_resolution)

    # The min detectable car is a 100x100 rectangle
    MIN_AREA = 10000
    
    # Set the back ground frame to nothing
    background = None


    for cnt in tqdm(range(frames_cnt), position=0, leave=True):

        ret, frame = cap.read() 

        if ret:
            if cars_detection:

                if background is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    # if the first frame is None, initialize it
                    background = gray
                    continue
                    
                cars = detect_moving_objects(frame, background)
                for car in cars:
                    
                    (x, y, w, h) = car
                    car = frame[y:y+h,x:x+w,:]
                    if debug: print('Car size', car.shape)                        
                    if detect_LP_fn != None:
                        # Pass the cropped car image to LPD
                        car_LP, LPs = detect_LP_fn(car, debug)
                        # Put back the LP patch in the original frame
                        frame[y:y+h,x:x+w,:] = car_LP

                    if show_cars_bbox == 1: # Just rectangle
                        cv2.rectangle(frame, (x, y), (x + w, y + h), SCALAR_RED, 10)
                    elif show_cars_bbox == 2: # Oriented rectangle
                        draw_oriented_bbox(frame, car)
            
            elif detect_LP_fn != None:
                frame, LPs = detect_LP_fn(frame, debug)
                

            if debug: plot_img(frame)
            out.write(cv2.resize(frame, output_video_resolution))
        else:
            print('no video')
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        

    cap.release()
    out.release()
    print('Video is ready at: ', video_output_file)

