import argparse
from network import *
from predict_load import load_categories

# Define the argparse to read out the arguments from the command line
parser = argparse.ArgumentParser(description="Argument Parser for prediction")

parser.add_argument(action='store',
                    dest='directory',
                    help='Stores the path/to/image')

parser.add_argument(action='store',
                    dest='checkpoint',
                    help='Stores the path/to/checkpoint')

parser.add_argument('--top_k', action='store',
                    type=int,
                    dest='top_k',
                    help='Return top K most likely classes')

parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    help='Use mapping of categories to real names')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use gpu for inference')

command_line_inputs = parser.parse_args()

network = Network()
network.load(command_line_inputs.checkpoint, command_line_inputs.gpu)
[probabilities], classes = network.predict(command_line_inputs.directory, command_line_inputs.top_k)
cat_to_name = load_categories(command_line_inputs.category_names)
# Output result
for i, j in zip(classes, probabilities):
	print("{} = {}%".format(cat_to_name[i], j*100))
