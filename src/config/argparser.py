import argparse
import os
import yaml

def argparser():

	actions = ["dl_data", "seq_data", "train", "predict"]

	parser = argparse.ArgumentParser()
	parser.add_argument("action", choices=actions, help='Action to execute', type=str)
	parser.add_argument('-f', dest='parameters_file', metavar="config_yaml_filepath", help='Path to parameters file', required=True)
	arg = parser.parse_args()

	with open(arg.parameters_file, 'r') as stream:
		params = yaml.safe_load(stream)

	return arg.action, params
