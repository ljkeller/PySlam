import numpy as np
import argparse
from collections import deque
import cv2
from matplotlib import pyplot as plt

# Our classes
from Extractor import *
from Preprocessor import *

MIN_MATCH_COUNT = 30

# Useful statistics for post-video analysis
def printStatistics(*, totalFrames, lowFeatureFrames):
    print(f'Total frames: {totalFrames}.')
    print(f'Low feature Frames: {lowFeatureFrames}.')
    print(f'Total % of low-feature frames: %{lowFeatureFrames/totalFrames*100:.2f}.')

def createArgumentParser():
    # Make argument parser for PySlam
    parser = argparse.ArgumentParser(description='Python SLAM implementation.')

    # required arguments
    parser.add_argument('image_path', type=str, help='Absolute or relative path of the video file')

    # optional arguments
    parser.add_argument('-s', '--scale_percent', nargs='?', default=50, type=int, help='An integer percentage of preprocessor image scaling (default=50)')
    parser.add_argument('-m', '--feature_matches', nargs='?', default=10, type=int, help='An integer representing how many feature matches to visualiz (default 10)')
    parser.add_argument('-f', '--num_features', nargs='?', default=500, type=int, help='An integer representing the maximum number of features to retain (default 500)')
    parser.add_argument('-t', '--score_type', nargs='?', default=0, type=int, help='An integer representing the feature detection type of 0:Harris, or 1:Fast (default:0)')

    return parser

def getProgramArguments():
    return createArgumentParser().parse_args()

def main():

    # Take command line arguments for various params
    args = getProgramArguments()

    # track recent data
    frameDeque = deque(maxlen=2)
    kpDeque = deque(maxlen=2)
    desDeque = deque(maxlen=2)
    
    # Create feature extractor, capture stream, and preprocessor
    fe = Extractor(args.num_features)
    cap = cv2.VideoCapture(args.image_path)
    pp = Preprocessor(scalePercent=args.scale_percent)

    # For post-video statistics
    totalFrames = lowFeatureFrames = 0

    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            printStatistics(totalFrames=totalFrames, lowFeatureFrames=lowFeatureFrames)
            exit()
        totalFrames += 1
    
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

            # Get perspective transform
            if len(matches)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kpDeque[0][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kpDeque[1][m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w,d = frameDeque[0].shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                # Overlay perspective transform so its visible how camera is moving through environment
                frameDeque[1] = cv2.polylines(frameDeque[1],[np.int32(dst)],True,255,3, cv2.LINE_AA)
            else:
                lowFeatureFrames += 1
                matchesMask = None
            
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)

            inlierMatches = cv2.drawMatches(frameDeque[0],kpDeque[0],frameDeque[1],kpDeque[1],matches,None,**draw_params)
            cv2.imshow("Inlier Matches", inlierMatches)

            # TODO: Develop 2 models, one for planar and one for non planar scene. Calculate idea model for given situation & use it

            # Draw first 10 matches
            matchImage = cv2.drawMatches(frameDeque[0], kpDeque[0], frameDeque[1], kpDeque[1], matches[:args.feature_matches], None, flags=2)
            cv2.imshow("Matches", matchImage)
            cv2.imshow("Last Frame", frameDeque[1])

        if cv2.waitKey(30) & 0xFF == ord('q'):
            printStatistics(totalFrames=totalFrames, lowFeatureFrames=lowFeatureFrames)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
