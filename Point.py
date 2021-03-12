class Point():
    """
    A class used to represent map points

    ...

    Attributes
    ----------
    pos3d : tuple
        3-D position in the world coordinate system
    viewingDirection : np.array
        the mean unit vector of all its viewing directions (the rays that join the point with 
        the optical center of the keyframes that observer it)
    descriptor : descriptor
        Associated ORB descriptor
    dmax : int
        Max distance at which the point can be observed according to limits of ORB features
    dmin : int
        Min distance at which the point can be observed according to limits of ORB features

    """
    def __init__(self, pos3D, viewingDirection, descriptor, dmax, dmin):
        self.pos3D = pos3D
        self.viewingDirection = viewingDirection
        self.descriptor = descriptor
        self.dmax = dmax
        self.dmin = dmin
