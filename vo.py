import numpy as np
import sys
from collections import deque
import cv2
from matplotlib import pyplot as plt

from Extractor import *

def preproc(frame, scale_percent=50):
    # grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # downscale
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    dsize = (w, h)

    return cv2.resize(img, dsize)


def main():
    
    if len(sys.argv) == 2:
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        print("Not enough arguments")
        exit(1)

    # track recent data
    frameDeque = deque(maxlen=2)
    kpDeque = deque(maxlen=2)
    desDeque = deque(maxlen=2)
    
    # Create feature extractor
    fe = Extractor()

    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            continue
    
        img = preproc(frame)
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

            # Sort descriptors in order of their distance
            #matches = sorted(matches, key = lambda x:x.queryIdx)

            # Draw first 10 matches
            matchImage = cv2.drawMatches(frameDeque[0], kpDeque[0], frameDeque[1], kpDeque[1], matches[:10], None, flags=2)
            cv2.imshow("Matches", matchImage)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
