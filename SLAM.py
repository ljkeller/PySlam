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
import OpenGL.GL as gl
import pangolin

# Make mapping class that inherits from Thread
class Mapper(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.poses = []

    def draw_trajectory(self, trajectory):
        # Draw lines
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 0.0)

        pangolin.DrawLine(trajectory)   # consecutive

        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawLines(
            trajectory, 
            trajectory + np.random.randn(len(trajectory), 3), 
            point_size=5)   # separate

    def draw_keyframe(self, pose, coord):
        # Create base translation matrix
        pose = np.identity(4)
        print(pose[:3, 3])
        
        # Set dx, dy, dz to random values (coordinate changes)
        pose[:3, 3] = np.random.randn(3)

        self.poses.append(pose)

        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawCameras(self.poses, 0.5, 0.75, 0.8)


    def test(self):
        print(test)
    # Override run method, this will now run in parrallel upon thread.start()
    def run(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define projection and initial ModelView matrix

        # Define Projection and initial ModelView matrix
        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(scam)

        # Create Interactive View in window
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        dcam.SetHandler(handler)

        
        trajectory = [[20, -6, 6]]
        for i in range(300):
            trajectory.append(trajectory[-1] + np.random.random(3)-0.5)
        trajectory = np.array(trajectory)

        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)
            
            # Render OpenGL Cube
            #pangolin.glDrawColouredCube(0.1)

            # Draw Point Cloud
            """
            points = np.random.random((10000, 3)) * 3 - 4
            gl.glPointSize(1)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(points)
            """

            # Draw Point Cloud
            """
            points = np.random.random((10000, 3))
            colors = np.zeros((len(points), 3))
            colors[:, 1] = 1 -points[:, 0]
            colors[:, 2] = 1 - points[:, 1]
            colors[:, 0] = 1 - points[:, 2]
            points = points * 3 + 1
            gl.glPointSize(1)
            pangolin.DrawPoints(points, colors)
            """

            self.draw_trajectory(trajectory)
            self.draw_keyframe(None, None)
            """
            # Draw lines
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 0.0)
            pangolin.DrawLine(trajectory)   # consecutive
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawLines(
                trajectory, 
                trajectory + np.random.randn(len(trajectory), 3), 
                point_size=5)   # separate

            """

            # Draw camera
            """
            pose = np.identity(4)
            pose[:3, 3] = np.random.randn(3)
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)
            """

            # Draw boxes
            """
            poses = [np.identity(4) for i in range(10)]
            for pose in poses:
                pose[:3, 3] = np.random.randn(3) + np.array([5,-3,0])
            sizes = np.random.random((len(poses), 3))
            gl.glLineWidth(1)
            gl.glColor3f(1.0, 0.0, 1.0)
            pangolin.DrawBoxes(poses, sizes)
            """

            pangolin.FinishFrame()

MIN_MATCH_COUNT = 30

# Useful statistics for post-video analysis
def printStatistics(*, totalFrames, lowFeatureFrames, totalKeyframes):
    print(f'Total frames: {totalFrames}.')
    print(f'Low feature Frames: {lowFeatureFrames}.')
    print(f'Total % of low-feature frames: {lowFeatureFrames/totalFrames*100:.2f}%.')
    print(f'Total keyframes: {totalKeyframes}.')

def createArgumentParser():
    # Make argument parser for PySlam
    parser = argparse.ArgumentParser(description='Python SLAM implementation.')

    # required arguments
    #parser.add_argument('image_path', type=str, help='Absolute or relative path of the video file')

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

    cap = Endpoint() if args.rpi_stream else cv2.VideoCapture(args.image_path)
    
    pp = Preprocessor(scalePercent=args.scale_percent)
    
    map_system = Mapper()
    map_system.start()

    # For post-video statistics
    totalFrames = lowFeatureFrames = 0
    framesSinceKeyframeInsert = 0

    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            printStatistics(totalFrames=totalFrames, lowFeatureFrames=lowFeatureFrames, totalKeyframes=len(keyframes))
            exit()

        # Iterate statistics variables
        totalFrames += 1
        framesSinceKeyframeInsert += 1
    
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
                # Identifies matches between keypoints deques and puts them into their respective lists
                # TODO: Change old frame to src and newframe to dst
                src_pts = np.float32([ kpDeque[0][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kpDeque[1][m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                
                # Homography model is good estimator for planar scenes
                pose, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

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

                # TODO: Develop 2 models, one for planar and one for non planar scene. Calculate idea model for given situation & use it

                # TODO: update keyframe insertion criteria to be more reflective of ORB
                if framesSinceKeyframeInsert >= 20:
                    # TODO: Add intrinsics
                    keyframes.append(Keyframe(pose=pose, features=features, intrinsics=None))
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


        if cv2.waitKey(30) & 0xFF == ord('q'):
            printStatistics(totalFrames=totalFrames, lowFeatureFrames=lowFeatureFrames, totalKeyframes=len(keyframes))
            break

    map_system.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
