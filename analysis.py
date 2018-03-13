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

from inception import Inception3, inception_v3
from torch.autograd import Variable

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

class Edge_inception(Inception3):
	def forward(self , x):
		print('overridden')
		if self.transform_input:
			x = x.clone()
			x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
			x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
			x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
		# 299 x 299 x 3
		x = self.Conv2d_1a_3x3(x)
		# 149 x 149 x 32
		x = self.Conv2d_2a_3x3(x)
		# 147 x 147 x 32	
		x = self.Conv2d_2b_3x3(x)
		# 147 x 147 x 64
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		# 73 x 73 x 64
		x = self.Conv2d_3b_1x1(x)
		# 73 x 73 x 80
		x = self.Conv2d_4a_3x3(x)
		# 71 x 71 x 192
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		return x


class ServerInception(Inception3):
	def forward(self, x):
		print('Server overridden')
		# 35 x 35 x 192
		x = self.Mixed_5b(x)
		# 35 x 35 x 256
		x = self.Mixed_5c(x)
		# 35 x 35 x 288
		x = self.Mixed_5d(x)
		# 35 x 35 x 288
		x = self.Mixed_6a(x)
		# 17 x 17 x 768
		x = self.Mixed_6b(x)
		# 17 x 17 x 768
		x = self.Mixed_6c(x)
		# 17 x 17 x 768
		x = self.Mixed_6d(x)
		# 17 x 17 x 768
		x = self.Mixed_6e(x)
		# 17 x 17 x 768
		if self.training and self.aux_logits:
			aux = self.AuxLogits(x)
		# 17 x 17 x 768
		x = self.Mixed_7a(x)
		# 8 x 8 x 1280
		x = self.Mixed_7b(x)
		# 8 x 8 x 2048
		x = self.Mixed_7c(x)
		# 8 x 8 x 2048
		x = F.avg_pool2d(x, kernel_size=8)
		# 1 x 1 x 2048
		x = F.dropout(x, training=self.training)
		# 1 x 1 x 2048
		x = x.view(x.size(0), -1)
		# 2048
		x = self.fc(x)
		# 1000 (num_classes)
		if self.training and self.aux_logits:
			return x, aux
		return x


def read_first_frame(FLAGS):
	print('Reading Frame')
	vidcap = cv.VideoCapture(FLAGS.video_file)
	success, image = vidcap.read()
	count = 0
	success = True
	success, image = vidcap.read()
	print('Read a new frame: ', success)
	# cv2.imwrite(FLAGS.frame_file+"frame%d.jpg" % count, image)  # save frame as JPEG file
	# cv2.imwrite('C:/Users/Kevin/PycharmProjects/MobileNetPrototype/inception_analysis_v1/frames/frame.jpg', image)
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
		print('Read a new frame: ', success)
		cv.imwrite("frames/frame%d.jpg" % count, image)  # save frame as JPEG file
		count += 1
	return count

def load_in_frame(path_to_image):
	# # v1
	# p = transforms.Resize(299)
	# #img = Image.open('/mnt/c/Users/Kevin/Documents/College/ECE4/FYP/python_test_code/photos/frame0.jpg')
	# img = Image.open(FLAGS.image_file)
	# print('Reading frame at: ',FLAGS.image_file)
	# img = p(img)
	# img = ToTensor()(img).unsqueeze(0)
	# # return img
	# print('Type of original image: ', type(img))

	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
		transforms.Resize(299),
		transforms.ToTensor(),
		normalize
	])
	img = Image.open(path_to_image)
	img = preprocess(img)
	img = img.unsqueeze(0)
	# print('Type of new image:  ',type(img))
	return img


def read_in_frame_number(frameNumber):

	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
		transforms.Resize(299),
		transforms.ToTensor(),
		normalize
	])
	img = Image.open('frames/frame%d.jpg' %frameNumber)
	img = preprocess(img)
	img = img.unsqueeze(0)
	# print('Type of new image:  ', type(img))
	return img

# Get the range of values of image to pick
# A good value for quantization
def get_range_of_values():
	print('Getting rang of values' )


def encode(array, max_num=8, num_bins=64):
	print('Encoding..')
	arr = array.data.numpy().squeeze(0)
	# print('original array: ', arr)

	arr = (np.minimum(arr,max_num)/8)*num_bins
	arr = np.round(arr)

	# print('Array output: ', arr)
	# print('Encoded siz of the array: ', sys.getsizeof(arr))

	return arr


def compute_delta(previous_array, current_array, delta_value):
	print('Computing deltas')
	delta_array = previous_array - current_array
	itop, jtop, ktop = delta_array.shape
	for i in range(itop):
		for j in range(jtop):
			for k in range(ktop):
				if(delta_array[i][j][k] in range(-delta_value,delta_value)):
					delta_array[i][j][k] = 0
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
	# print('decoded array: ', server_input)

	incept = torchvision.models.inception_v3(pretrained=True)
	incept.eval()
	fc_out = ServerInception.forward(self=incept, x=server_input)

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
		print('ground truth in top5')
		return True
	else:
		print('Classification incorrect')
		return False