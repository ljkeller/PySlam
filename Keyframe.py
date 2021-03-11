class Keyframe:
    def __init__(self, pose, features, intrinsics):
        # 3x3 matrix of transformation for now
        self.pose = pose

        # Features present in this keyframe
        self.kps = features['kps']
        self.descriptors = features['des']
        
        # focal length & principle point
        self.intrinsics = intrinsics
