import torch
import torch.nn as nn 
import numpy as np 
import torchvision

class CNNNetwork(nn.Module):
	def __init__(self, out_features=4):
		super(CNNNetwork, self).__init__()
		self.model = torchvision.models.densenet169(pretrained=True)
		self.final_fcn = nn.Linear(1000, 4)

		for i, param in enumerate(self.model.features.parameters()):
			if(i < 400):
				param.requires_grad = False

	def forward(self, x):
		return self.final_fcn(self.model(x))

if __name__ == '__main__':
	model = CNNNetwork()
	x = torch.randn((5,3,224,224))

	# for k, v in model.model.features._modules.items():
	# 	print(k, v)

	y = model(x)
	print(sum(p.numel() for p in model.parameters() if p.requires_grad))

	print(y.shape)

