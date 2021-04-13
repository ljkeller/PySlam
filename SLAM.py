import numpy as np
import argparse
from threading import Thread
from collections import deque
import cv2
from matplotlib import pyplot as plt

# Our classes
from Extractor import *
from Preprocessor import *
from Keyframe import *
from Endpoint import Endpoint

# Mapping
from Map import *

MIN_MATCH_COUNT = 30

# Useful statistics for post-video analysis
def printStatistics(*, totalFrames, lowFeatureFrames, totalKeyframes):
    print(f'Total frames: {totalFrames}.')
    print(f'Low feature Frames: {lowFeatureFrames}.')
    print(f'Total % of low-feature frames: {lowFeatureFrames/totalFrames*100:.2f}%.')
    print(f'Total keyframes: {totalKeyframes}.')

def convert_world2pangolin(point):
    t = np.array([[0,0,-1],[1,0,0],[0,1,0]])
    return np.matmul(t, np.asarray(point))

def createArgumentParser():
    # Make argument parser for PySlam
    parser = argparse.ArgumentParser(description='Python SLAM implementation.')

    # optional arguments, require additional args
    parser.add_argument('-i', '--image_path', nargs='?', default = "", \
            type=str, help='Absolute or relative path of the video file')
    parser.add_argument('-s', '--scale_percent', nargs='?', default=50, \
            type=int, help='An integer percentage of preprocessor image scaling (default=50)')
    parser.add_argument('-m', '--feature_matches', nargs='?', default=10, \
            type=int, help='An integer representing how many feature matches to visualiz (default 10)')
    parser.add_argument('-f', '--num_features', nargs='?', default=500, \
            type=int, help='An integer representing the maximum number of features to retain (default 500)')
    parser.add_argument('-t', '--score_type', nargs='?', default=0, \
            type=int, help='An integer representing the feature detection type of 0:Harris, or 1:Fast (default:0)')

    # optional boolean action arguments, dont require additional args
    parser.add_argument('-d', '--debug', action='store_true', \
            help='An argument that enables debug mode, which prints information & displays several screens')
    parser.add_argument('-r', '--rpi_stream', action='store_true', \
            help='An argument specifying to run PiSlam in client/server configuration (default port:30000)')

    return parser

# Simple wrapper to increase readability
def getProgramArguments():
    return createArgumentParser().parse_args()

def main():

    # Take command line arguments for various params
    args = getProgramArguments()

    # track recent data
    # TODO: change features to tuple? named_tuple('Feature', ['keypoint', 'descriptor'] & update code
    frameDeque = deque(maxlen=2)
    kpDeque = deque(maxlen=2)
    desDeque = deque(maxlen=2)
    keyframes = []
    
    # Create feature extractor, capture stream, and preprocessor
    fe = Extractor(args.num_features)

    # Link with RPI depending upon program arguments
    cap = Endpoint() if args.rpi_stream else cv2.VideoCapture(args.image_path)
    
    pp = Preprocessor(scalePercent=args.scale_percent)
    
    # Instantiate MAP & start mapping thread
    map_system = Mapper()
    map_system.start()

    # For post-video statistics
    totalFrames = lowFeatureFrames = 0
    framesSinceKeyframeInsert = 0

    acummulatingPose = np.identity(3)

    test = np.array([1,1,1])
    test = np.transpose(test)

    while(cap.isOpened()):

        ret, frame = cap.read()
        # TODO: catch sigint
        if not ret:
            printStatistics(totalFrames=totalFrames, \
                    lowFeatureFrames=lowFeatureFrames, totalKeyframes=len(keyframes))
            exit()

        # Iterate statistics variables
        totalFrames += 1
        framesSinceKeyframeInsert += 1
    
        img = pp.preprocess(frame)

        # Create Camera Intrinsics matrix for 3d-2d mapping
        H, W = img.shape
        focal_len = 100
        K = np.array([[focal_len/W, 0,           W//2, 0],
                      [0,           focal_len/H, H//2, 0],
                      [0,           0,           1, 1]])

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
                # Identifies matches between keypoints deques and puts them into their respective lists
                # TODO: Change old frame to src and newframe to dst
                src_pts = np.float32([ kpDeque[0][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kpDeque[1][m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                
                # Homography model is good estimator for planar scenes
                pose, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

                ret1, rot, tran, _ = cv2.decomposeHomographyMat(pose, K[0:3, 0:3])

                # TODO identify valid translation

                # Recover global & local transformation data
                F, maskf = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)

                E, maske = cv2.findEssentialMat(src_pts, dst_pts, K[0:3, 0:3], cv2.RANSAC, 0.9, 2)
                r1, r2, t = cv2.decomposeEssentialMat(E)

                if t[0] <0:
                	t = -t

                correct_translation = t

                # Two views - general moteion, general structure
                # 1. Esimate essential/fundamental mat
                # 2. Decompose the essential mat
                # 3. Impose positive depth requirement (up to 4 sols)
                # 4. Recover 3d structure

                # Note fundamental mat is sensitive to all points lying on same plane
                
                # Keep only inliers
                src_pts = src_pts[mask.ravel() == 1]
                dst_pts = dst_pts[mask.ravel() == 1]

                # Accumulate pose transformation to track global transform over time
                c1 = np.concatenate((acummulatingPose, np.array([map_system.system_coord]).T), axis=1)
                acummulatingPose = np.matmul(pose, acummulatingPose)
                system_plus_translation = np.array([map_system.system_coord+correct_translation]).T
                c2 = np.concatenate((acummulatingPose,system_plus_translation[0]), axis=1)
                #c1 = np.dot(K[:3,:3], c1)

                #c2 = np.zeros((3,4))
                #c2[:3,:3] = acummulatingPose
                #c2[:3,3] = np.transpose(-correct_translation)
                #c2 = np.dot(K[:3,:3], c2)
                
                points_4d = map_system.convert2D_4D(src_pts, dst_pts, c1, c2)
                mask_4d = np.abs(points_4d[:,3]) > .005
                points_new4d = points_4d[mask_4d]
                points_4d /= points_4d[3]
                mask_4d = points_4d[:,2]>0
                points_4d = points_4d[mask_4d]
                points_3d = points_4d[:,:3]
                points_3d = [convert_world2pangolin(np.transpose(point)) for point in points_3d]
                
                # Matrix of 3d points
                a1 = np.matmul(pose, test)
                a1 = a1/np.linalg.norm(a1)
                map_system.q.append((convert_world2pangolin(a1), points_3d))

                map_system.cur_pose = acummulatingPose

                # TODO: Get essential matrix here, then recover pose, find which model has min error
                #reval, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

                # Make array in which x_i == 1 -> x_i feature is a match between src & dst points
                matchesMask = mask.ravel().tolist()

                h,w,d = frameDeque[0].shape

                # Maps ROI to corners of image before performing perspective transform. New pose
                # of ROI is representative of transformations being performed on camera between frames
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,pose)


                # Overlay perspective transform so its visible how camera is moving through environment
                poseDeltaImage = cv2.polylines(frameDeque[0].copy(),[np.int32(dst)],True,255,3, cv2.LINE_AA)

                # TODO: Develop 2 models, one for planar and one for non planar scene. 
                # Calculate idea model for given situation & use it

                # TODO: update keyframe insertion criteria to be more reflective of ORB
                if framesSinceKeyframeInsert >= 20:
                    # TODO: Add intrinsics
                    keyframes.append(Keyframe(pose=pose, features=features, intrinsics=K))
                    framesSinceKeyframeInsert = 0

            else:
                lowFeatureFrames += 1
                matchesMask = None
            
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)

            # Inlier matches are those decided acceptable by RANSAC
            inlierMatches = cv2.drawMatches(frameDeque[0],kpDeque[0],\
                    frameDeque[1],kpDeque[1],matches,None,**draw_params)

            # Draw first args.feature_matches amount of matches
            matchImage = cv2.drawMatches(frameDeque[0], kpDeque[0], \
                    frameDeque[1], kpDeque[1], matches[:args.feature_matches], None, flags=2)
            
            if args.debug:
                cv2.imshow("Inlier Matches", inlierMatches)
                cv2.imshow("Matches", matchImage)

            cv2.imshow("Current frame with delta pose", poseDeltaImage)


        # Escape on <esc>
        if cv2.waitKey(30) == 27:
            printStatistics(totalFrames=totalFrames, lowFeatureFrames=lowFeatureFrames, \
                    totalKeyframes=len(keyframes))
            map_system.stop()
            break

    map_system.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
