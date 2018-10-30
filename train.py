import argparse

from train_load import *
from network import Network

# Define the argparse to read out the arguments from the command line
parser = argparse.ArgumentParser(description="Argument Parser for prediction")

parser.add_argument(action='store',
                    dest='directory',
                    help='Defines the data directory for training data')

parser.add_argument('--arch', action='store',
                    default='vgg16',
                    dest='arch',
                    help='Defines the architecture for training')

parser.add_argument('--save_dir', action='store',
                    default='checkpoints/',
                    dest='save_dir',
                    help='Defines the directory to save checkpoints')

parser.add_argument('--learning_rate', action='store',
                    default=0.001,
                    type=float,
                    dest='learning_rate',
                    help='Defines the learning rate')

parser.add_argument('--hidden_units', action='store',
                    default=512,
                    type=int,
                    dest='hidden_units',
                    help='Defines the hidden units for the network')

parser.add_argument('--epochs', action='store',
                    default=10,
                    type=int,
                    dest='epochs',
                    help='Defines the epochs for training the network')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use gpu for inference')

command_line_inputs = parser.parse_args()

if command_line_inputs.arch in AVAILABLEMODELS:
	input_size = AVAILABLEMODELS[command_line_inputs.arch]
	arch = command_line_inputs.arch
else:
	raise Exception("Specified arch is not available")

# Use load_data to create generator objects for training, validation and testing
trainloader, validloader, testloader, class_to_idx, batch_size = load_data(command_line_inputs.directory)

# Build Network
network = Network(input_size, command_line_inputs.hidden_units, command_line_inputs.learning_rate, arch,
																		command_line_inputs.epochs, command_line_inputs.gpu)

print("Building Model...", end='')
network.build_model()
print(" done")
print("Start Training...")
network.train(trainloader, validloader)
print("Finished Training".center(80, '-'))
print("Start Testing...")
network.test(testloader)
print("Finished Testing".center(80, '-'))
print("Saving Network to", command_line_inputs.save_dir)
network.save(command_line_inputs.save_dir, class_to_idx, batch_size)
print("Finished Saving".center(80, '-'))
