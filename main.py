import argparse
import os
from video import process_video
import char
import morpho




def main(args=None):
    if args.detect_LP_fn == 1:
        detect_LP_fn = char.detect_LP
    elif args.detect_LP_fn == 2:
        detect_LP_fn = morpho.detect_LP
    else:
        detect_LP_fn = None

    process_video(args.video_file, 
                  args.video_output_file, 
                  frames_cnt=args.frames_cnt,
                  cars_detection=args.cars_detection,
                  show_cars_bbox=args.show_cars_bbox,
                  detect_LP_fn=detect_LP_fn,
                  debug=args.debug
                  )

if __name__ == '__main__':


    parser = argparse.ArgumentParser(prog='License Plate Detector (LPD)')
    parser.add_argument('--video_file', help='the path to the input video')
    parser.add_argument('--video_output_file', help='the path to the output video')
    parser.add_argument('--frames_cnt', help='The desired number of frames to process. If None the whole video is processed.', 
                        type=int, default=-1)
    parser.add_argument('--cars_detection', help='LPD will work on the car cropped image or whole image.', 
                        type=bool, default=True)                        
    parser.add_argument('--show_cars_bbox', help='0: no cars shown, 1: bbox shown, 2: oriented bbox shown', 
                        type=int, default=0)
    parser.add_argument('--detect_LP_fn', help='The desired LPD method, 0: char, 1: morphology. If not passed no LPs are detected.', 
                        type=int, default=-1)
    parser.add_argument('--debug', help='Set to True to see intermediate outputs and debug logs', 
                        type=bool, default=False)                         
                                              
    
    args = parser.parse_args()

    main(args)
