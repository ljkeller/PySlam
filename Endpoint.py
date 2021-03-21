# This file is a class stream wrapper meant to read in a video stream from a raspberry pi and return the frames.
# This file works in conjunction with piVideoStream.py.
# The ip address and connection port should be passed to constructor.

# Endpoint.py
import io
import socket
import struct
from PIL import Image
import numpy as np
import cv2
import math

class Endpoint():
    def __init__(self, host='', port=30000):
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Listen on all host, port
        self.server_socket.bind((host, port))

        # Queue at most 1 connection
        self.server_socket.listen(1)

        # Accept a single connection and make a file-like object out of it
        self.connection = self.server_socket.accept()[0].makefile('rb')

        # Width and height of the video stream from the PI
        self.imW, self.imH = 640, 480

        # Construct a stream to hold the image data and read the image
        # data from the connection. Probably are better ways, but this works for now
        self.image_stream = io.BytesIO()

        print("Successfully Established connection with Pi")
    
    # Returns (true, frame) on success
    #         (false, None) on failure
    def read(self):
        # Read the length of the image as a 32-bit unsigned int.
        image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
        if image_len == False:
            return (False, None)

        # TODO: make thread-safe queue later
        self.image_stream = io.BytesIO()
        image = None
        try:
            # reading jpeg image from binary stream
            self.image_stream.write(self.connection.read(image_len))
            # reconstruct image with pillow
            image = Image.open(self.image_stream)
        except:
            # if reading raw images: yuv or rgb
            image = Image.frombytes('L', (self.imW, self.imH), self.image_stream.read())

        # Rewind the stream
        self.image_stream.seek(0)
        
        #Convert to cv2 usable image
        npImage = np.array(image)
        return (True, npImage)

    # Detects if socket stream still open by peeking and catching common
    def isOpened(self):
        """
        try:
            # this will try to read bytes without blocking and also without removing them from buffer (peek only)
            data = self.connection.recv(16, socket.MSG_DONTWAIT | socket.MSG_PEEK)
            if len(data) == 0:
                return True
        except BlockingIOError:
            return True  # socket is open and reading from it would block
        except ConnectionResetError:
            return False  # socket was closed for some other reason
        except Exception as e:
            logger.exception("unexpected exception when checking if a socket is closed")
            return False
"""
        return True

    # Wraps up sockets, call this before closing program.
    # Would put in constructor, but have to replicate cv2 behaviour
    def release(self):
        print("release\n")
        self.connection.close()
        self.server_socket.close()
