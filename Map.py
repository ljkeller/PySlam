#Standard libs
import numpy as np
from threading import Thread
import math
import cv2

# Mapping
import OpenGL.GL as gl
import pangolin

# Make mapping class that inherits from Thread
class Mapper(Thread):
    def __init__(self):
        self.stopped = False
        Thread.__init__(self)
        
        # TODO: share keyframe map with main thread
        self.poses = []
        self.points = []
        self.path = []

        self.q = []
        self.feature_queue = []


        self.cur_pose = None

        # Generator for next position
        #self.system_coord = self.generate_trajectory()
        self.system_coord = 0

    def init_window(self, name='System Mapping', w=640, h=480):
        pangolin.CreateWindowAndBind(name, w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
    
    def set_projection_and_model_view(self):
        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        self.handler = pangolin.Handler3D(self.scam)

    def setup_interactive_display(self):
        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    # To stop Mapper run() loop
    def stop(self):
        self.stopped = True

    # Helper function to evaluate if main is attempting to join threads
    def continue_mapping(self):
        return not pangolin.ShouldQuit() and not self.stopped

    # Constant velocity model, will of course update this
    # Generator function that will calculate next coordinate
    def generate_trajectory(self):
        # Starting at origin
        coord = np.zeros(3)
        while not pangolin.ShouldQuit():
            yield coord
            coord = coord + np.random.rand(1,3)[0] # Add some random amount to previous

    # Returns current coordinate of system from global perspective
    def current_system_coordinate(self):
        if self.path:
            return self.path[-1]
        else:
            return None
    
    # Draws the path of the system through the world
    def draw_trajectory(self):
        if len(self.path) > 1:
            # Draw lines
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 0.0)

            #pangolin.DrawLine(self.path)   # consecutive
            pangolin.DrawLine(np.array(self.path))

    # Draws a keyframe with given pose at given coordinate
    def draw_keyframe(self, pose, coord):
        # Create base translation matrix
        if pose is None:
            pose = np.identity(4)

        # Fix later, pi is not 1.8
        angle_radian = 1.8
        y = [[math.cos(angle_radian),0,math.sin(angle_radian)],[0,1,0],[-math.sin(angle_radian),0,math.cos(angle_radian)]]
        pose[:3,:3] = np.matmul(y, pose[:3,:3])
        
        # Set dx, dy, dz to random values (coordinate changes)
        if coord is None:
            pose[:3, 3] = np.random.randn(3)
        else:
            pose[:3, 3] = coord

        if self.cur_pose is not None:
            #pose[0:3, 0:3] = np.matmul(self.cur_pose, np.identity(3))
            pass

        self.poses.append(pose)

        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawCameras(self.poses, 0.5, 0.75, 0.8)

    def convert2D_4D(self, points_2d_source, points_2d_dest, pose_1, pose_2):
        points = cv2.triangulatePoints(pose_1[:3], pose_2[:3], points_2d_source, points_2d_dest)
        return np.transpose(points)


    # Draws points at new_points with their ABSOLUTE coordinate
    # Appends new_points to points list, TODO: use our data structures
    def draw_point_cloud(self, new_points, size=2):
        if new_points is None:
            return
        new_points = (new_points + np.transpose(self.system_coord)).tolist()
        #new_points = new_points + np.transpose(self.system_coord)
        self.points += new_points
        #print(new_points)
        gl.glPointSize(size)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(self.points)

    # Override run method, this will now run in parrallel upon thread.start()
    def run(self):
        # Not initializing window until mapping start() called
        self.init_window()
        self.set_projection_and_model_view()
        self.setup_interactive_display()

        # TODO: Probably block on draw() call here
        # Continue until <esc>
        while self.continue_mapping():
            if len(self.q) <= 0:
                continue

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)
            
            translation, new_points = self.q.pop()
            self.system_coord = self.system_coord + translation            
            
            self.path.append(np.transpose(self.system_coord)[0])

            self.draw_trajectory()
            self.draw_keyframe(pose=None, coord=np.transpose(self.system_coord))
            self.draw_point_cloud(new_points)

            pangolin.FinishFrame()


