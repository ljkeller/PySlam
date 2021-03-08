import numpy as np
import cv2

class Preprocessor:
    def __init__(self, scalePercent=50):
        self.imgScalePercent = scalePercent

    def preprocess(self, frame):
        # grayscale
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # downscale
        w = int(img.shape[1] * self.imgScalePercent / 100)
        h = int(img.shape[0] * self.imgScalePercent / 100)
        dsize = (w, h)

        return cv2.resize(img, dsize)


