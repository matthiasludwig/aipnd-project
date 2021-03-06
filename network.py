import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

from train_load import AVAILABLEMODELS
from predict_load import process_image


class Network:
	def __init__(self, input_size=25088, hidden_units=512, lr=0.001, arch='vgg16', epochs=10, gpu=True):
		# Initialize all values passed to the object
		self.input_size = input_size
		self.hidden_units = hidden_units
		self.learning_rate = lr
		self.arch = arch
		if self.arch == 'vgg16':
			self.model = models.vgg16(pretrained=True)
			if self.input_size != AVAILABLEMODELS['vgg16']:
				raise ValueError("self.input units is not compatible to chosen architecture!")
		elif self.arch == 'alexnet':
			self.model = models.alexnet(pretrained=True)
			if self.input_size != AVAILABLEMODELS['alexnet']:
				raise ValueError("self.input units is not compatible to chosen architecture!")
		elif self.arch == 'densenet161':
			self.model = models.densenet161(pretrained=True)
			if self.input_size != AVAILABLEMODELS['densenet161']:
				raise ValueError("self.input units is not compatible to chosen architecture!")
		self.epochs = epochs
		self.gpu = gpu
		self.class_to_idx = None
		self.criterion = nn.NLLLoss()
		self.optimizer = None

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
			('fc2', nn.Linear(self.hidden_units, 102)),
			('softmax', nn.LogSoftmax(dim=1))
		]))
		self.model.classifier = classifier
		self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)

	def validation(self, validloader):
		valid_loss = 0
		accuracy = 0

		for inputs, labels in validloader:
			inputs, labels = inputs.to(self.device), labels.to(self.device)

			output = self.model.forward(inputs)
			valid_loss = self.criterion(output, labels).item()

			ps = torch.exp(output)
			equality = (labels.data == ps.max(dim=1)[1])
			accuracy += equality.type(torch.FloatTensor).mean()

		return valid_loss, accuracy

	def train(self, trainloader, validloader):
		steps = 0
		print_every = 50

		self.model.to(self.device)

		for e in range(self.epochs):
			running_loss = 0
			for inputs, labels in trainloader:
				steps += 1

				self.model.train()  # To ensure that the model in in train mode

				inputs, labels = inputs.to(self.device), labels.to(self.device)

				self.optimizer.zero_grad()

				outputs = self.model.forward(inputs)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()

				if steps % print_every == 0:
					# Put model in eval mode
					self.model.eval()

					with torch.no_grad():
						valid_loss, accuracy = self.validation(validloader)

					print("Epoch: {}/{}.. ".format(e + 1, self.epochs),
											"Training Loss: {:.3f}.. ".format(running_loss / print_every),
											"Valid Loss: {:.3f}.. ".format(valid_loss / len(validloader)),
											"Valid Accuracy: {:.3f}".format(accuracy / len(validloader)))

					running_loss = 0

					self.model.train()

	def test(self, testloader):
		# Do test on the testloader
		test_loss = 0
		accuracy = 0

		self.model.eval()

		with torch.no_grad():
			for inputs, labels in testloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)

				output = self.model.forward(inputs)
				test_loss = self.criterion(output, labels).item()

				ps = torch.exp(output)
				equality = (labels.data == ps.max(dim=1)[1])
				accuracy += equality.type(torch.FloatTensor).mean()

		print("Test-Loss: {}\n".format(test_loss / len(testloader)),
								"Test-Accuracy: {}".format(accuracy / len(testloader)))

	def save(self, save_dir, class_to_idx, batch_size):
		# Save the checkpoint
		save_loc = save_dir + 'checkpoint.pth'

		checkpoint = {
			'input_size': self.input_size,
			'batch_size': batch_size,
			'state_dict': self.model.classifier.state_dict(),
			'class_to_idx': class_to_idx,
			'output_size': 102,
			'classifier': self.model.classifier,
			'arch': self.arch
		}

		torch.save(checkpoint, save_loc)

	def load(self, filepath, gpu):
		checkpoint = torch.load(filepath)

		# Restoring Model with model.arch
		if checkpoint['arch'] == 'vgg16':
			self.model = models.vgg16(pretrained=True)
		elif checkpoint['arch'] == 'alexnet':
			self.model = models.alexnet(pretrained=True)
		elif checkpoint['arch'] == 'densenet161':
			self.model = models.densenet161(pretrained=True)
		self.model.classifier = checkpoint['classifier']
		self.gpu = gpu
		self.class_to_idx = checkpoint['class_to_idx']

		# Restoring model.classifier.state_dict()
		self.model.classifier.load_state_dict(checkpoint['state_dict'])

	def predict(self, image_path, topk):
		# Predict the class (or classes) of an image using a trained deep learning model.
		# Inverting class_to_idx. Adapted from: https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
		idx_to_class = {v: k for k, v in self.class_to_idx.items()}
		# Check if gpu should be used
		if self.gpu:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # If gpu used, check for cuda
		else:
			self.device = 'cpu'
		# Predict image with preprocessing
		image = torch.FloatTensor([process_image(image_path)])
		image = image.to(self.device)
		self.model = self.model.to(self.device)
		self.model.eval()
		with torch.no_grad():
			prediction = self.model.forward(image)

		# Save probabilities and classes
		probabilities = torch.exp(prediction).topk(topk)[0]
		probabilities = probabilities.to('cpu')  # Copy back to CPU to ensure numpy conversion later
		classes = np.array(torch.exp(prediction).topk(topk)[1])
		classes = [idx_to_class[c] for c in classes[0, :]]  # classes.shape = (1,5)

		return probabilities.numpy(), classes
