import numpy as np
import cv2

class Extractor:
    """
    A class used for feature extraction and matching

    ...

    Attributes
    ----------
    orb : cv2.ORB
        ORB object used to detect features from frames
    bfm : cv2.BFMatcher
        Brute force matcher that will match descriptors from different sets

    Methods
    -------

    extract(self, frame)
        finds keypoints and descriptors within a frame and returns them

    match(self, des1, des2)
        finds matching features between two sets of descriptors
    """
    # scoreType 
    #           0 : Harris 1 
    #           1 : FAST 

    def __init__(self, numFeatures=500, scoreType=0):

        # create orb object with default params
        self.orb = cv2.ORB_create(nfeatures=numFeatures, scoreType=scoreType)

        # create BFMatcher object used to brute-force match ORB descriptors
        # uses norm_hamming for ORB specifically
        # crossCheck=True filters outlier matches very well in highly-featured environments. Effectively k=1 in knnMatch
        self.bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract(self, frame):
        # Find the keypoints, then compute the descriptors
        kp = self.orb.detect(frame, None)
        pts = cv2.goodFeaturesToTrack(frame, 3000, qualityLevel=0.01, minDistance=7)

        if pts is None:
            return {"kps":None, "des":None}

        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=40) for f in pts]
        print(kps)

        kp, des = self.orb.compute(frame, kps)
        return {"kps":kp, "des":des}

    def match(self, des1, des2):
        return self.bfm.match(des1, des2)
