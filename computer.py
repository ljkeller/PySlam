# server.py
import io
import socket
import struct
from PIL import Image
import numpy as np
import cv2
import math

def drawMethod(cv2Image):
	#Method to draw a cv2 image to the screen
	cv2.imshow('Window', cv2.cvtColor(cv2Image, cv2.COLOR_BGR2RGB))
	cv2.waitKey(1)

def pillowToNPArray(pillowImage):
	return np.array(pillowImage)

def stream():
	# Adapted from https://jmlb.github.io/robotics/2017/11/22/picamera_streaming_video/
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.bind(('', 8000))
	server_socket.listen(0)
	# Accept a single connection and make a file-like object out of it
	connection = server_socket.accept()[0].makefile('rb')
	W, H = 28, 28

	try:
		while True:
			# Read the length of the image as a 32-bit unsigned int.
			image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

			if not image_len:
				break
			# Construct a stream to hold the image data and read the image
			# data from the connection
			image_stream = io.BytesIO()
			image = None
			try:
				# reading jpeg image
				image_stream.write(connection.read(image_len))
				image = Image.open(image_stream)
			except:
				# if reading raw images: yuv or rgb
				image = Image.frombytes('L', (W, H), image_stream.read())
			# Rewind the stream
			image_stream.seek(0)
			
			#Convert to cv2 usable image
			newIm = pillowToNPArray(image)

			drawMethod(newIm)	
			
	finally:
		connection.close()
		server_socket.close()
		



if __name__ == '__main__':
	stream()
