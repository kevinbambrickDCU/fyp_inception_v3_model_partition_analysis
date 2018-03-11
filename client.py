#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import torchvision
import argparse
import sys
import  analysis

from torch.autograd import Variable

FLAGS = None

def main():
	analysis.read_first_frame(FLAGS)
	img = analysis.load_in_frame(FLAGS)

	incept = torchvision.models.inception_v3(pretrained=True)
	incept.eval()
	edge_out = analysis.Edge_inception.forward(self=incept, x=Variable(img))

	encoded_edge_output = analysis.encode(edge_out)
	print('Sent to server')
	send(encoded_edge_output)


def send(data):
	# Create a TCP/IP socket
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	# Connect the socket to the port where the server is listening
	server_address = ('localhost', 10000)
	print(sys.stderr, 'connecting to %s port %s' % server_address)
	sock.connect(server_address)

	try:
		message = (data)
		sock.sendall(message)

		# Look for the response
		amount_received = 0
		amount_expected = len(message)

		while amount_received < amount_expected:
			data = sock.recv(1024)
			amount_received += len(data)
	finally:

		print(sys.stderr, 'closing socket')
		sock.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--image_file',
		type=str,
		default='frames/frame0.jpg',
		help='Absolute path to image file.'
	)
	parser.add_argument(
		'--frame_file',
		type=str,
		default='frames/',
		help='Absolute path to the folder storing the frames'
	)
	parser.add_argument(
		'--video_file',
		type = str,
		default='videos/test_vid.mp4',
		help='Absolute path to the folder storing the video to be analysed'
	)
	parser.add_argument(
		'--num_top_predictions',
		type = int,
		default = 5,
		help = 'Number of predictions to show'
	)

	FLAGS, unparsed = parser.parse_known_args()
	main()