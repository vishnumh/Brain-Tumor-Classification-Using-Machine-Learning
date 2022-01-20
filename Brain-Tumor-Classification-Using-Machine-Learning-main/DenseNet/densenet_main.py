import torch
from model import CNNNetwork
from cnn_dataset import MyDataset
from transforms import *
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2

def calculate_accuracy(pred, label):
	_, pred = torch.max(pred, axis=1)
	return torch.sum(pred == label).item()


def train():
	custom_transforms = transforms.Compose([RandomHorizontalFlip(),
	                                       RandomVerticalFlip(),
	                                       RandomHorizontalVerticalFlip(),
	                                       Resize(),
	                                       ToTensor()])

	dataset = MyDataset(transform=custom_transforms)
	model = CNNNetwork()
	loss_fn = nn.CrossEntropyLoss().cuda()
	epoch = 20

	if torch.cuda.is_available():
	    device = torch.device("cuda")    
	print(device)

	model = model.to(device)
	model.train()
	print("Model Loaded")
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

	for i in range(1,epoch+1):
	    data = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
	    correctly_predicted = 0
	    total_image_no = 0
	    total_loss = 0
	    
	    for batch_no, batch_data in enumerate(data):
	        image_input = batch_data['input_image'].to(device)
	        label = batch_data['label'].to(device)
	        label = torch.squeeze(label, axis=1)

	        pred = model(image_input)
	        loss = loss_fn(pred, label) #.float().cuda()
	        
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	        
	        total_loss += loss.item()
	        correctly_predicted += calculate_accuracy(pred, label)
	        total_image_no += image_input.shape[0]

	        
	    print("Epoch:", i, "Epoch_Loss:", total_loss / total_image_no, "Accuracy", correctly_predicted / total_image_no)

	torch.save(model.state_dict(), "desnsenet_weights.pth", )


def test():
	custom_transforms = transforms.Compose([RandomHorizontalFlip(),
	                                       RandomVerticalFlip(),
	                                       RandomHorizontalVerticalFlip(),
	                                       Resize(),
	                                       ToTensor()])

	dataset = MyDataset(filename="labelled_test.csv", transform=custom_transforms, data_type='Testing')
	model = CNNNetwork()

	if torch.cuda.is_available():
	    device = torch.device("cuda")    

	model.load_state_dict(torch.load("/home/akshay/ml_project/desnsenet_weights.pth"))
	model = model.to(device)
	model.eval()

	data = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

	correctly_predicted = 0
	total_image_no = 0

	for batch_no, batch_data in enumerate(data):
	    image_input = batch_data['input_image'].to(device)
	    label = batch_data['label'].to(device)
	    label = torch.squeeze(label, axis=1)

	    pred = model(image_input)
	    correctly_predicted += calculate_accuracy(pred, label)
	    total_image_no += image_input.shape[0]

	print("Test Accuracy:", correctly_predicted / total_image_no)

if __name__ == "__main__":
	train()
	test()