'''
1. 程式名稱：test.py
2. 程式內容：Test the Performance of Lane Detection
    (1) Grayscale
    (2) Use HSV to find the yellow&white mask
    (2) Canny Edge Detection
    (3) Hough Transform
    (4) Group Left and Right Lines
    (5) Fit and Draw

可修改的部份以加入了
gamma correction, CLAHE
'''

from Land_Detection.land_detection import *
import argparse

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-v", "--video", required=True, help="path to input video")
    args = vars(ap.parse_args())
    test_video(args["video"])

    # ap.add_argument("-i", "--image", required=True, help="path to input image")
    # args = vars(ap.parse_args())
    # test_img(args["image"])

if __name__ == '__main__':
    main()

