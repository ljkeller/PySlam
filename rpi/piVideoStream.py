# This file is to be ran on a raspberry pi with the official pi camera.
# This file works in conjunction with server.py.
# The ip address and connection port should be modified to your use.

# client.py
import io
import socket
import struct
import time
import picamera

# Adapted from https://jmlb.github.io/robotics/2017/11/22/picamera_streaming_video/

def write_img_to_stream(stream):
    connection.write(struct.pack('<L', stream.tell()))
    connection.flush()
    stream.seek(0)  #seek to location 0 of stream_img
    connection.write(stream.read())  #write to file
    stream.seek(0)
    stream.truncate()


def gen_seq():
    stream = io.BytesIO()
    while True:
        yield stream
        write_img_to_stream(stream)

# Connect a client socket to server_ip:8000
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Ip address for server, current set as self
ip_address = "127.0.0.1"
client_socket.connect( ( ip_address, 8000 ) )
# Make a file-like object out of the connection
connection = client_socket.makefile('wb')

if __name__ == '__main__':
    try:
        with picamera.PiCamera() as camera:
            # Set resolution of PI camera
            camera.resolution = (640,480)
            # Adjust framerate to correspond with camera resolution
            camera.framerate = 20
            # Create sequence of image from camera
            camera.capture_sequence(gen_seq(), "jpeg", use_video_port=True)
        connection.write(struct.pack('<L', 0))
    finally:
        connection.close()
        client_socket.close()
