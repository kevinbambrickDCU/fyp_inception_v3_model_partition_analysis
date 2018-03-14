import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision
import requests
import cv2 as cv
from PIL import Image
import numpy as np
import sys
import json
import math

from inception import Inception3, inception_v3
from torch.autograd import Variable

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

class SplitComputation(Inception3):
	def forward(self, x,start = 0,end = None):
		print('overridden')

		if self.transform_input and (start == 0):
			x = x.clone()
			x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
			x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
			x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

		def my_pool(x):
			return F.max_pool2d(x, kernel_size=3, stride=2)

		def my_pool2(x):
			return F.avg_pool2d(x, kernel_size=8)

		def my_dropout(x):
			return F.dropout(x, training=self.training)

		def my_view(x):
			return x.view(x.size(0), -1)

		layers = [
			self.Conv2d_1a_3x3,
			self.Conv2d_2a_3x3,
			self.Conv2d_2b_3x3,
			my_pool,
			self.Conv2d_3b_1x1,
			self.Conv2d_4a_3x3,
			lambda x: F.max_pool2d(x, kernel_size=3, stride=2),
			self.Mixed_5b,
			self.Mixed_5c,
			self.Mixed_5d,
			self.Mixed_6a,
			self.Mixed_6b,
			self.Mixed_6c,
			self.Mixed_6d,
			self.Mixed_6e,
			self.Mixed_7a,
			self.Mixed_7b,
			self.Mixed_7c,
			my_pool2,
			my_dropout,
			my_view,
			self.fc,
		]

		for layer in layers[start:end]:
			x = layer(x)
		return x


def read_first_frame(FLAGS):
	print('Reading Frame')
	vidcap = cv.VideoCapture(FLAGS.video_file)
	success, image = vidcap.read()
	count = 0
	success = True
	success, image = vidcap.read()
	print('Read a new frame: ', success)
	cv.imwrite('frames/' + "frame%d.jpg" % count, image)  # save frame as JPEG file
	count += 1


def read_in_all_frames(fileName):
	print('Reading Frames from: ',fileName)
	vidcap = cv.VideoCapture(fileName)
	success, image = vidcap.read()
	count = 0
	success = True
	while success:
		success, image = vidcap.read()
		print('Read a new frame: ', success, ' :', count)
		cv.imwrite("frames/frame%d.jpg" % count, image)  # save frame as JPEG file
		count += 1
	return count

def read_in_frame_per_second(fileName):
	cap = cv.VideoCapture(fileName)
	frameRate = cap.get(5)  # frame rate
	count = 0
	while (cap.isOpened()):
		frameId = cap.get(1)  # current frame number
		ret, frame = cap.read()
		if (ret != True):
			break
		if (frameId % math.floor(frameRate) == 0):
			filename = ("frames/frame%d.jpg" % count)
			cv.imwrite(filename, frame)
		count += 1
	cap.release()
	return count


def load_in_frame(path_to_image):
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
		transforms.Resize((299,299)),
		transforms.ToTensor(),
		normalize
	])
	img = Image.open(path_to_image)
	img = preprocess(img)
	img = img.unsqueeze(0)

	return img


def read_in_frame_number(frameNumber):
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
		transforms.Resize((299,299)),
		transforms.ToTensor(),
		normalize
	])
	img = Image.open('frames/frame%d.jpg' %frameNumber)
	img = preprocess(img)
	img = img.unsqueeze(0)

	return img


# Get the range of values of image to pick
# A good value for quantization
#WIP
def get_range_of_values():
	print('Getting rang of values' )


def encode(array, max_num=8, num_bins=64):
	print('Encoding..')
	arr = array.data.numpy().squeeze(0)
	arr = (np.minimum(arr,max_num)/8)*num_bins
	arr = np.round(arr)

	return arr


def compute_delta(previous_array, current_array, delta_value):
	print('Computing deltas')
	delta_array = previous_array - current_array
	delta_array[abs(delta_array)>delta_value]=0
	print(delta_array)
	return delta_array


def decode_delta(previous_array, delta_array):
	print('decoding deltas')
	return previous_array - delta_array


def decode(array, max_num = 8, num_bins=64):
	print('Decoding')
	arr = (array.astype(np.float32)/num_bins)*max_num
	arr = np.expand_dims(arr, axis= 0)
	arr = torch.Tensor(arr)

	return Variable(arr)


def get_max_and_min(array):
	sort = array.data.numpy().argsort().squeeze(0)
	flat_arr = array.data.numpy().flatten()
	maxes = array.data.numpy().argmax()
	mins = array.data.numpy().argmin()
	print('Maxes: ',maxes)
	print('Mins: ',mins)
	print('Max: ', flat_arr[maxes])
	print('Min: ', flat_arr[mins])


def server_run(input):
	server_input = decode(input)

	incept = torchvision.models.inception_v3(pretrained=True)
	incept.eval()
	fc_out = SplitComputation.forward(self = incept, x= server_input,
									 start =7, end=None)

	sort = fc_out.data.numpy().argsort()
	#download labels
	# labels = {int(key): value for (key, value)
	# 		  in requests.get(LABELS_URL).json().items()}

	#read labels from file
	labels = {int(key): value for (key, value)
			  in json.load(open('config/labels.json')).items()}

	match = False
	print(labels[fc_out.data.numpy().argmax()])
	# num = FLAGS.num_top_predictions + 1
	for i in range(1, 6):
		print('Number ', i, ': ', labels[sort[0][-i]])
		if(sort[0][-i] == 105):
			match = True

	if(match):
		print('Ground truth in top-5')
		return True
	else:
		print('Classification incorrect')
		return False