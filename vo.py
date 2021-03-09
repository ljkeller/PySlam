import numpy as np
import argparse
from collections import deque
import cv2
from matplotlib import pyplot as plt

# Our classes
from Extractor import *
from Preprocessor import *

# TODO: Implement some statistics once video is done displaying
def print_statistics():
    pass

def main():

    # Take command line arguments for various params
    parser = argparse.ArgumentParser(description='Python SLAM implementation.')
    parser.add_argument('image_path', type=str, help='Absolute or relative path of the video file')
    parser.add_argument('-s', '--scale_percent', nargs='?', default=50, type=int, help='An integer percentage of preprocessor image scaling (default=50)')
    parser.add_argument('-m', '--feature_matches', nargs='?', default=10, type=int, help='An integer representing how many feature matches to visualiz (default 10)')
    args = parser.parse_args()

    # track recent data
    frameDeque = deque(maxlen=2)
    kpDeque = deque(maxlen=2)
    desDeque = deque(maxlen=2)
    
    # Create feature extractor, capture stream, and preprocessor
    fe = Extractor()
    cap = cv2.VideoCapture(args.image_path)
    pp = Preprocessor(scalePercent=args.scale_percent)

    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            print_statistics()
            exit()
    
        img = pp.preprocess(frame)
        features = fe.extract(img)
        if features['kps'] is None or features['des'] is None:
            continue

        # record image, keypoints, and descriptors between frames
        kpDeque.appendleft(features['kps'])
        desDeque.appendleft(features['des'])
        kpImg = cv2.drawKeypoints(img, kpDeque[0], None, color=(0,255,0), flags=0)
        frameDeque.appendleft(kpImg)

        # match between frames
        if len(frameDeque) > 1:

            # match descriptors
            matches = fe.match(desDeque[0], desDeque[1])

            # Draw first 10 matches
            matchImage = cv2.drawMatches(frameDeque[0], kpDeque[0], frameDeque[1], kpDeque[1], matches[:args.feature_matches], None, flags=2)
            cv2.imshow("Matches", matchImage)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
