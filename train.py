import argparse

# Define the argparse to read out the arguments from the command line
parser = argparse.ArgumentParser(description="Argument Parser for prediction")

parser.add_argument(action='store',
                    dest='directory',
                    help='Stores the data directory for training')

parser.add_argument('--arch', action='store',
                    dest='arch',
                    help='Defines the architecture for training')  # TODO: Input possible values

parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='Defines the directory to save checkpoints')

parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    help='Defines the learning rate')

parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    help='Defines the hidden units for the network')

parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    help='Defines the epochs for training the network')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='use_gpu',
                    help='Use gpu for inference')

command_line_inputs = parser.parse_args()