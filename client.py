#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import torchvision
import argparse
import sys
import analysis
import inception
import json

from torch.autograd import Variable

FLAGS = None

def main():
	# classify_one_image('frames/frame45.jpg')
	classify_video()
	# classify_video_without_splitting()

def classify_video():
	print('Classifying Video frame by fame')
	#dumps frames into file
	# number_of_frames = analysis.read_in_all_frames(FLAGS.video_file)
	number_of_frames = 299
	PREVIOUS_ARRAY = None
	for i in range(number_of_frames):
		img = analysis.read_in_frame_number(i)

		incept = torchvision.models.inception_v3(pretrained=True)
		incept.eval()
		edge_out = analysis.Edge_inception.forward(self=incept, x=Variable(img))

		encoded_edge_output = analysis.encode(edge_out)

		#delta encoding not implimented in server yet
        #
		if PREVIOUS_ARRAY is not None:
			print('Current Arry: ', (encoded_edge_output))
			input_to_compute_deltas = encoded_edge_output
			print('previous: ', (PREVIOUS_ARRAY))
			delta_encoded_edge_output = analysis.compute_delta(PREVIOUS_ARRAY,input_to_compute_deltas, 2)
			print('delta: ',(delta_encoded_edge_output))
			send(delta_encoded_edge_output)
		else:
			send(encoded_edge_output)

		# send(encoded_edge_output.tobytes())
		# send(encoded_edge_output)

		PREVIOUS_ARRAY = encoded_edge_output

def classify_video_without_splitting():
	number_of_frames = 299
	# number_of_frames = analysis.read_in_all_frames(FLAGS.video_file)
	passCount = 0
	failCount = 0
	for i in range(number_of_frames):
		img = analysis.read_in_frame_number(i)

		incept = torchvision.models.inception_v3(pretrained=True)
		incept.eval()

		fc_out = inception.Inception3.forward(self = incept, x = Variable(img))
		sort = fc_out.data.numpy().argsort()
		# download labels
		# labels = {int(key): value for (key, value)
		# 		  in requests.get(LABELS_URL).json().items()}

		# read labels from file
		labels = {int(key): value for (key, value)
				  in json.load(open('config/labels.json')).items()}

		match = False
		print(labels[fc_out.data.numpy().argmax()])
		# num = FLAGS.num_top_predictions + 1
		for i in range(1, 6):
			print('Number ', i, ': ', labels[sort[0][-i]])
			if (sort[0][-i] == 105):
				match = True

		if (match):
			print('ground truth in top5')
			passCount += 1
		else:
			print('Classification incorrect')
			failCount += 1

		print('Total checked: ', passCount + failCount)
		print('Number of correct classifictaions: ', passCount)
		print('Number Failed: ', failCount)


def classify_one_image(path_to_image):
	print('Classifying one image')
	# analysis.read_first_frame(FLAGS)
	img = analysis.load_in_frame(path_to_image)

	incept = torchvision.models.inception_v3(pretrained=True)
	incept.eval()
	edge_out = analysis.Edge_inception.forward(self=incept, x=Variable(img))

	encoded_edge_output = analysis.encode(edge_out)

	send(encoded_edge_output)

def send(data):
	data = data.astype('int8').tobytes()
	print('Data being sent to server')
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