#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket
import torchvision
import argparse
import sys
import analysis
import inception
import json
import numpy as np
import pickle
import os
import random
import shutil

from torch.autograd import Variable
from dahuffman import HuffmanCodec

FLAGS = None
use_delta = True

path_to_imagnet = '../imagnet/val/'
huff_freq = None

def main():
	delete_previous_frames(FLAGS.frame_file)
	# classify_one_image('frames/frame1.jpg')
	# classify_one_image('../imagnet/val/n01739381/ILSVRC2012_val_00022816.JPEG')
	# classify_one_image('../imagnet/val/n01847000/ILSVRC2012_val_00000415.JPEG')
	classify_video(FLAGS.video_file, write=True)
	# classify_video_without_splitting(FLAGS.video_file, 105)
	# classify_video_without_splitting("videos/n02509815/panda_5.mp4", 388)
	# classify_video('videos/n03452741/piano_2.mp4', write=True)


	# classify_on_random_images(path_to_imagnet,1)

def delete_previous_frames(path_to_frames):
	for the_file in os.listdir(path_to_frames):
		file_path = os.path.join(path_to_frames, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
			# elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)


def classify_video(path_to_file, write =False):
	print('Classifying Video frame by fame')
	#dumps frames into file
	# number_of_frames = analysis.read_in_all_frames(path_to_file)
	# number_of_frames = 299
	fps , number_of_frames = analysis.get_fps_and_number_of_frames(path_to_file)
	# number_of_frames = analysis.read_in_frame_per_second(path_to_file)
	PREVIOUS_ARRAY = None

	incept = torchvision.models.inception_v3(pretrained=True)
	incept.eval()

	for i in range(number_of_frames):
		# img = analysis.read_in_frame_number_from_file(i)

		img = analysis.read_in_frame_from_video(path_to_file, i*fps, write=write)


		# 7 is cut at F.max_pool2d
		edge_out = analysis.SplitComputation.forward(self = incept,
													 x= Variable(img),
													 start = 0,
													 end=7)

		if PREVIOUS_ARRAY is not None and use_delta is True:
			input_to_compute_deltas = edge_out.data.numpy().squeeze(0)
			delta_edge_output = analysis.compute_delta(PREVIOUS_ARRAY,input_to_compute_deltas, 0.1)

			delta_encoded_edge_output = analysis.encode(delta_edge_output,min_num=-8,
														 max_num=8,num_bins=60)
			# print("data being sent: ",delta_encoded_edge_output)
			send(delta_encoded_edge_output)
			PREVIOUS_ARRAY = PREVIOUS_ARRAY - analysis.decode(delta_encoded_edge_output).squeeze(0)

			# analysis.train_huff_tree(delta_encoded_edge_output, 'delta_hist', write_to_json= True)

		else:
			input_to_encoder = edge_out.data.numpy().squeeze(0)
			encoded_edge_output = analysis.encode(input_to_encoder,min_num=-8,
												   max_num=8,num_bins=60)
			send(encoded_edge_output)
			# analysis.train_huff_tree(encoded_edge_output)
			PREVIOUS_ARRAY = analysis.decode(encoded_edge_output).squeeze(0)


def classify_on_random_images(path_to_data_set, number_of_images_to_check):
	print('classifying images at: ', path_to_data_set)

	cats = json.load(open(FLAGS.cat_json))
	imagnet_folder = os.listdir(path_to_imagnet)
	num_of_folders = len(imagnet_folder)

	print(num_of_folders)

	for i in range(number_of_images_to_check):
		folder = random.choice(imagnet_folder)
		img_folder = path_to_imagnet+folder
		images_list = os.listdir(img_folder)
		image = random.choice(images_list)
		full_path_to_image = img_folder+'/'+image
		print('Classifying from: ', full_path_to_image)
		classify_one_image(full_path_to_image)


def classify_one_image(path_to_image):
	print('Classifying one image')
	# analysis.read_first_frame(FLAGS)
	img = analysis.load_in_image(path_to_image)

	incept = torchvision.models.inception_v3(pretrained=True)
	incept.eval()

	edge_out= analysis.SplitComputation.forward(self =incept, x = Variable(img),
												start=0, end=7)

	input_to_encoder = edge_out.data.numpy().squeeze(0)
	encoded_edge_output = analysis.encode(input_to_encoder,min_num=-8,
										  max_num=8,num_bins=64)

	print('shape of data: ', encoded_edge_output.shape)

	send(encoded_edge_output)


def classify_video_without_splitting(path_to_file, class_number):
	# number_of_frames = 299
	fps, number_of_frames = analysis.get_fps_and_number_of_frames(path_to_file)
	print(fps, number_of_frames)
	# number_of_frames = analysis.read_in_frame_per_second(path_to_file)

	incept = torchvision.models.inception_v3(pretrained=True)
	incept.eval()
	passCount = 0
	failCount = 0
	failed_frames ={}
	for i in range(number_of_frames):

		try:
			img = analysis.read_in_frame_from_video(path_to_file, i * fps, write=True)
		except Exception as e:
			break

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
			if (sort[0][-i] == class_number):
				match = True

		if (match):
			print('ground truth in top5')
			passCount += 1
		else:
			print('Classification incorrect')
			failCount += 1
			failed_frames[failCount-1] =failCount+passCount

		print('Total checked: ', passCount + failCount)
		print('Number of correct classifictaions: ', passCount)
		print('Number Failed: ', failCount)
	print('failed frames: ',failed_frames)
	print('percentage of passed: ', (passCount/(failCount+passCount))*100)


def send(data):
	print('shape: ',data.shape)
	print('Type being sent: ', type(data))
	# data = data.astype('int8').tobytes()


	# code for huffman
	# arr = data.astype('int8')
	# codec  = HuffmanCodec.from_data(arr.flatten())
	# encoded = codec.encode(arr.flatten())
	# print('size of data being sent: ', len(encoded))
	# data = (encoded, codec)
	# data = pickle.dumps(data)

	# new code for huffman
	arr = data.astype('int8')
	with open('huffman_encoding_config/delta_hist.pickle', 'rb') as handle:
		delta_hist = pickle.load(handle)
	codec = HuffmanCodec.from_data(delta_hist)
	encoded  = codec.encode(arr.flatten())
	data = encoded

	# print(codec.get_code_table())

	#new code for rle does not work
	# data = data.astype('int8')
	# starts, lengths, values = rle.rlencode(data.flatten())
	# data = zip(starts, lengths, values)
	# data = np.array(list(data)).astype('int8').tobytes()

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
		'--cat_json',
		type = str,
		default = 'config/categories.json',
		help = 'Path to data set of images'
	)

	FLAGS, unparsed = parser.parse_known_args()
	main()