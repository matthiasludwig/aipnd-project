import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F


class Network:
	def __init__(self, input_size, hidden_units, lr, arch, epochs, gpu):
		# Initialize all values passed to the object
		self.input_size = input_size
		self.hidden_units = hidden_units
		self.learning_rate = lr
		if arch == 'vgg16':
			self.model = models.vgg16(pretrained=True)
		elif arch == 'alexnet':
			self.model = models.alexnet(pretrained=True)
		elif arch == 'densenet161':
			self.model = models.densenet161(pretrained=True)
		self.epochs = epochs
		self.gpu = gpu
		self.criterion = nn.NLLLoss()
		self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)

		# Check if gpu should be used
		if self.gpu:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # If gpu used, check for cuda
		else:
			self.device = 'cpu'

	def build_model(self):
		# Substitute the model's classifier
		for param in self.model.parameters():
			param.requires_grad = False

		classifier = nn.Sequential(OrderedDict([
			('fc1', nn.Linear(self.input_size, self.hidden_units)),
			('relu1', nn.ReLU()),
			('dropout1', nn.Dropout(0.5)),
			('fc4', nn.Linear(self.hidden_units, 102)),
			('softmax', nn.LogSoftmax(dim=1))
		]))
		self.model.classifier = classifier
