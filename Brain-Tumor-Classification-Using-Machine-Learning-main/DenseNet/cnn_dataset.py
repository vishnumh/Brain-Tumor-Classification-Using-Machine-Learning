import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import pandas as pd
import os
import cv2
from PIL import Image


class MyDataset(Dataset):
	def __init__(self, filename="labelled.csv", transform=None, data_type="Training", base_path="/home/akshay/ml_project/archive/"):
		self.transform = transform
		self.file_list = pd.read_csv(filename)

		self.x = self.file_list['filename'] 
		self.y = self.file_list['label']
		self.base_path = base_path + data_type
	

	def __getitem__(self, idx):
		image_path = os.path.join(self.base_path, self.x[idx]) 
		input_image = Image.open(image_path)

		if input_image.mode != 'RGB':
			input_image = input_image.convert('RGB')

		label = self.y[idx]
		sample_data = {'input_image': input_image, 'label': label}
		sample_data = self.transform(sample_data)
		return sample_data

	def __len__(self):
		return len(self.x)


if __name__ == '__main__':
	d = MyDataset()