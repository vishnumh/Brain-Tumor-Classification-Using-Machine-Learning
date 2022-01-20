import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import cv2
import random
from PIL import Image, ImageOps


class RandomHorizontalFlip(object):
	def __call__(self, sample_data):
		input_image, label = sample_data['input_image'], sample_data['label']

		if random.random() < 0.3:
			input_image = ImageOps.flip(input_image)	

		return {'input_image': input_image, 'label':label}

class RandomVerticalFlip(object):
	def __call__(self, sample_data):
		input_image, label = sample_data['input_image'], sample_data['label']

		if random.random() < 0.3:
			input_image = ImageOps.mirror(input_image)	

		return {'input_image': input_image, 'label':label}


class RandomHorizontalVerticalFlip(object):
	def __call__(self, sample_data):
		input_image, label = sample_data['input_image'], sample_data['label']

		if random.random() < 0.3:
			input_image = ImageOps.mirror(input_image)	
			input_image = ImageOps.flip(input_image)

		return {'input_image': input_image, 'label':label}

class Resize(object):
	def __call__(self, sample_data):
		input_image, label = sample_data['input_image'], sample_data['label']
		input_image = input_image.resize((224,224))
		return {'input_image': input_image, 'label': label}

class ToTensor(object):
	def __call__(self, sample_data):
		input_image, label = sample_data['input_image'], sample_data['label']
		to_tensor = transforms.ToTensor()
		input_image = to_tensor(input_image)
		label = torch.Tensor([label])
		label = label.type(torch.LongTensor)
		input_image = input_image.type(torch.float32)

		return {'input_image': input_image, 'label': label}

if __name__ == '__main__':
	pass