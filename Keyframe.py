class Keyframe:
    """
    A class used to represent keyframes

    ...

    Attributes
    ----------
    pose : np.array
        Camera pose, which is rigit body transformation that transforms points from the world to the camera coordinate system
    intrinsics :
        Camera intrinsics, including focal length and principal point
    kps : keypoints
        Keypoints from image
    des : descriptors
        Descriptors for keypoints
    """
    def __init__(self, pose, features, intrinsics):
        # 3x3 matrix of transformation for now
        self.pose = pose

        # Features present in this keyframe
        self.kps = features['kps']
        self.descriptors = features['des']
        
        # focal length & principle point
        self.intrinsics = intrinsics
