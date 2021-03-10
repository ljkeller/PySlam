import numpy as np
import cv2

class Extractor:
    # scoreType 
    #           0 : Harris 1 
    #           1 : FAST 

    def __init__(self, numFeatures=500, scoreType=0):

        # create orb object with default params
        self.orb = cv2.ORB_create(nfeatures=numFeatures, scoreType=scoreType)

        # create BFMatcher object used to brute-force match ORB descriptors
        # uses norm_hamming for ORB specifically
        self.bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract(self, frame):
        # Find the keypoints, then compute the descriptors
        kp = self.orb.detect(frame, None)
        kp, des = self.orb.compute(frame, kp)
        return {"kps":kp, "des":des}

    def match(self, des1, des2):
        return self.bfm.match(des1, des2)
