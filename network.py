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
		# Do validation on the test set
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

	def save(self, save_dir):
		# Save the checkpoint
		save_loc = save_dir + 'checkpoint.pth'

		checkpoint = {
			'input_size': self.model.classifier[0].in_features,
			# 'batch_size': trainloader.batch_size,  #TODO
			'state_dict': self.model.classifier.state_dict(),
			# 'class_to_idx': train_data.class_to_idx,  # TODO
			'output_size': 102,
			'classifier': self.model.classifier
		}

		torch.save(checkpoint, save_loc)