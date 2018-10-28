import numpy as np
from network import *


def process_image(image):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
		returns an Numpy array
	'''
	# Scales the picture to 256, 256 via PIL.resize
	pil_image = Image.open(image)
	pil_image = pil_image.resize((256, 256))

	# Resize (Center-crop). Adapted from https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
	width, height = pil_image.size
	left = (width - 224) / 2
	top = (height - 224) / 2
	right = (width + 224) / 2
	bottom = (height + 224) / 2
	pil_image = pil_image.crop((left, top, right, bottom))

	# Convert to numpy array and normalize as described above
	np_image = np.array(pil_image) / 255
	np_image[:, :, 0] = (np_image[:, :, 0] - 0.485) / 0.229
	np_image[:, :, 1] = (np_image[:, :, 1] - 0.456) / 0.224
	np_image[:, :, 2] = (np_image[:, :, 2] - 0.406) / 0.225

	# Transpose the np array
	np_image = np_image.transpose((2, 0, 1))

	return np_image


def load_checkpoint(filepath):
	checkpoint = torch.load(filepath)

	# Restoring Model with model.classifier
	model = Network(checkpoint['input_size'], checkpoint['hidden_units'], )
	classifier = checkpoint['classifier']
	model.classifier = classifier
	# Restoring model.classifier.state_dict()
	model.classifier.load_state_dict(checkpoint['state_dict'])

	return model, checkpoint['class_to_idx']


#model, class_to_idx = load_checkpoint('checkpoints/checkpoint.pth')