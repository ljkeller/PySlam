import numpy as np
from collections import deque
import cv2
from matplotlib import pyplot as plt

def main():
    cap = cv2.VideoCapture("highway.mp4")

    # Create deques for tracking between monocular frames
    frameDeque = deque(maxlen=2)
    kpDeque = deque(maxlen=2)
    desDeque = deque(maxlen=2)

    # create orb object with default params
    orb = cv2.ORB_create()

    # create BFMatcher object used to brute-force match ORB descriptors
    # uses norm_hamming for ORB specifically
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while(cap.isOpened()):

        ret, frame = cap.read()
        gFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Find the keypoints, then compute the descriptors
        kp = orb.detect(gFrame, None)
        kp, des = orb.compute(gFrame, kp)

        gFrame2 = cv2.drawKeypoints(gFrame, kp, None, color=(0,255,0), flags=0)
        
        # record image, keypoints, and descriptors between frames
        frameDeque.appendleft(gFrame2)
        kpDeque.appendleft(kp)
        desDeque.appendleft(des)

        if len(frameDeque) > 1:

            # match descriptors
            matches = bf.match(desDeque[0], desDeque[1])

            # Sort descriptors in order of their distance
            #matches = sorted(matches, key = lambda x:x.distance)

            # Draw first 10 matches
            img3 = cv2.drawMatches(frameDeque[0], kpDeque[0], frameDeque[1], kpDeque[1], matches[:10], None, flags=2)
            cv2.imshow("Matches", img3)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    #cv2.imshow("doggo", img)
    #cv2.waitKey()

    #initiate STAR detector
    #orb = cv2.ORB_create()

    # find the keypoints
    #kp = orb.detect(img,None)

    # compute the descriptors with ORB
    #kp, des = orb.compute(img, kp)

    # draw only keypoints location, not size and orientation
    #img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    #plt.imshow(img2), plt.show()

if __name__ == "__main__":
    main()
