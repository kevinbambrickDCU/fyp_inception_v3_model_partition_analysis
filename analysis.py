import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision
import requests
import cv2 as cv
from PIL import Image
import argparse
import numpy as np 
import sys

from inception import Inception3, inception_v3
from torchvision.transforms import ToTensor
from torch.autograd import Variable

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

FLAGS = None

class Edge_inception(Inception3):
	def forward(self , x):
		print('overridden')
		# if self.transform_input:
		#           x = x.clone()
		#           x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
		#           x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
		#           x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
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

def read_first_frame():
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

def load_in_frame():
	p = transforms.Resize(299)
	#img = Image.open('/mnt/c/Users/Kevin/Documents/College/ECE4/FYP/python_test_code/photos/frame0.jpg')
	img = Image.open(FLAGS.image_file)
	print('Reading frame at: ',FLAGS.image_file)
	img = p(img)
	img = ToTensor()(img).unsqueeze(0)
	return img

# Get the range of values of image to pick 
# A good value for quantization
def get_range_of_values():
	print('Getting rang of values' )	

# Problemm in encoder numbers in array not being converted to uint8
# Not one clue why, encoder & decoder working otherwise
def encode(array, max_num=8, num_bins=128):
	arr = array.data.numpy().squeeze(0)
	print('Original size of num in array',sys.getsizeof(arr[0][0][0]))
	print('Original tye of num in array: ', type(arr[0][0][0]))
	itop, jtop, ktop = arr.shape
	for i in range(itop):
		for j in range(jtop):
			for k in range(ktop):
				x= arr[i][j][k]
				x = min(x,max_num) 
				x = x/max_num # Number in range 0 -> 1
				x = x*num_bins # Number in range 0 -> 64
				x= x.astype('uint8')
				# x = x.astype(float)
				# x = np.uint8(x)
				arr[i][j][k] = x
	test = arr[0][0][0]		
	print('encoder test num: ',test)
	print('Encoder test number type: ',type(test))	
	print('Encoder test number size: ',sys.getsizeof(test))
	print(sys.getsizeof(39.0))					
	return arr		

def decode(array, max_num = 8, num_bins=128 ):
	arr = array
	print(arr.shape)
	itop, jtop, ktop = arr.shape
	for i in range(itop):
		for j in range(jtop):
			for k in range(ktop):
				z = arr[i][j][k]
				z = z/num_bins
				z = z*max_num
				arr[i][j][k] = z
	arr = torch.Tensor(arr)	
	arr = arr.squeeze(0)		
	print(type(arr))
	print(arr.shape)
	print(arr.squeeze(0))
		
def get_max_and_min(array):
	sort = array.data.numpy().argsort().squeeze(0) 
	flat_arr = array.data.numpy().flatten()
	maxes = array.data.numpy().argmax()
	mins = array.data.numpy().argmin()
	print('Maxes: ',maxes)
	print('Mins: ',mins)

	print('Max: ', flat_arr[maxes])
	print('Min: ', flat_arr[mins])

def main():

	read_first_frame()
	img = load_in_frame()

	# edge run 
	incept = torchvision.models.inception_v3(pretrained=True)
	incept.eval()
	edge_out = Edge_inception.forward(self = incept, x = Variable(img))

	# print('Original output of Edge run ')
	# print(edge_out)

	arr = encode(edge_out)
	# print('Output of encoded edge: ')
	# print(arr)
	## END OF EDGE RUN ##

	print('Sent to server')
	## START OF SERVER RUN ## 
	server_input = decode(arr)
	#Server run 
	fc_out = ServerInception.forward(self = incept, x = edge_out)
	
	sort = fc_out.data.numpy().argsort()

	labels = {int(key):value for (key, value)
		in requests.get(LABELS_URL).json().items()}


	print(labels[fc_out.data.numpy().argmax()])
	num = FLAGS.num_top_predictions+1
	for i in range(1,num):
		print('Number ',i,': ',labels[sort[0][-i]])

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
		help = 'Number of perditions to show'
	)

	FLAGS, unparsed = parser.parse_known_args()
	main()