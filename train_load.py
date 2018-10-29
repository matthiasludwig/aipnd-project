import torch
from torchvision import transforms, datasets

AVAILABLEMODELS = {'alexnet': 9216,
                   'vgg16': 25088,
                   'densenet161': 2208}

def load_data(directory):
	train_dir = directory + '/train'
	valid_dir = directory + '/valid'
	test_dir = directory + '/test'

	train_transforms = transforms.Compose([transforms.Resize(224),
						transforms.RandomRotation(30),
						transforms.RandomResizedCrop(224),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize([0.485, 0.456, 0.406],
											[0.229, 0.224, 0.225])])
	valid_transforms = transforms.Compose([transforms.Resize(256),
	                                       transforms.CenterCrop(224),
	                                       transforms.ToTensor(),
	                                       transforms.Normalize([0.485, 0.456, 0.406],
	                                                            [0.229, 0.224, 0.225])])
	test_transforms = transforms.Compose([transforms.Resize(256),
	                                      transforms.CenterCrop(224),
	                                      transforms.ToTensor(),
	                                      transforms.Normalize([0.485, 0.456, 0.406],
	                                                           [0.229, 0.224, 0.225])])

	# Load the datasets with ImageFolder
	train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
	valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
	test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

	# Using the image datasets and the trainforms, define the dataloaders
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
	validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=False)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

	return trainloader, validloader, testloader, train_data.class_to_idx, trainloader.batch_size
