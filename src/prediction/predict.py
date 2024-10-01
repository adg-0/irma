from keras.optimizers import Adam
from keras.metrics import binary_accuracy, binary_crossentropy, sparse_categorical_accuracy, sparse_categorical_crossentropy

from model.model import get_model
from model.metrics import *

from data.data_sequencer import load_sequence_data

import numpy as np
import pandas as pd
from datetime import datetime


LOSS_DICT = {
	"sparse_categorical_crossentropy": sparse_categorical_crossentropy
}


LABEL_MAP = {
	(0, 1): 1,
	(1, 0): 0
}


# For each sequence, makes a DF, writes the label and predictions, and saves DF
def save_predictions(val, predictions):

	# One hot vector to 1D vector
	#predictions = np.array([LABEL_MAP[(round(a), round(b))] for (a, b) in predictions])

	for (x, y_true, y_pred) in zip(*val, predictions):
		df = pd.DataFrame(x)
		#df.set_index(1, drop=False, inplace=True)  # not really usefull
		
		df["class"] = np.nan
		df["pred_0"] = np.nan
		df["pred_1"] = np.nan

		df.loc[df.index[-1], "class"] = y_true
		df.loc[df.index[-1], "pred_0"] = y_pred[0]
		df.loc[df.index[-1], "pred_1"] = y_pred[1]

		token = df.loc[df.index[0], 0]

		df.to_csv(f"{PREDICTION_FOLDER}{token}-{datetime.timestamp(datetime.now())}.csv", index=False)


def split_results(l):
	return l[:-4], l[-4:]


def save_results(evaluation, metrics):
	with open(f"{LOGS_FOLDER}metrics.txt", 'a') as file:
		for i in range(len(evaluation)):
			file.write(metrics[i].__name__ + ": " + str(evaluation[i]) + "\n")

	for i in range(len(evaluation)):
		print(metrics[i].__name__ + ": ", evaluation[i])


def save_confusion_matrix(matrix):
	results_all = np.reshape(np.array(matrix), (2, 2))
	sum_ = np.sum(results_all, axis=1)  # sum by line
	sum_ = np.reshape(np.repeat(sum_, 2, 0), (2, 2))
	results_by_class = results_all / sum_
	results_by_class = np.around(100 * results_by_class, 2)
	# results_all = np.around(100 * results_all, 2)

	helper_df = pd.DataFrame(np.array([["TP", "FN"], ["FP", "TN"]]), ["+", "-"], ["+", "-"])
	results_all_df = pd.DataFrame(results_all, ["+", "-"], ["+", "-"])
	results_by_class_df = pd.DataFrame(results_by_class, ["+", "-"], ["+", "-"])

	with open(f"{LOGS_FOLDER}conf_matrix.txt", 'a') as file:
		file.write("Ground truth in line\n")
		file.write("Predicted in column\n")
		file.write(helper_df.to_string())
		file.write("\n\n")
		file.write("Overall results\n")
		file.write(results_all_df.to_string())
		file.write("\n\n")
		file.write("Results by class\n")
		file.write(results_by_class_df.to_string())

	print()
	print("Ground truth in line")
	print("Predicted in column")
	print(helper_df)
	print()
	print("Overall results")
	print(results_all_df)
	print()
	print("Results by class")
	print(results_by_class_df)


def predict(params):

	global LOGS_FOLDER
	global PREDICTION_FOLDER

	SEQUENCE_FOLDER = params["path"]["sequence_folder"]
	LOGS_FOLDER = params["path"]["logs_folder"]
	MODEL_PATH = f"{params['path']['model_folder']}{params['model']['id_name']}.h5"
	PREDICTION_FOLDER = params["path"]["prediction_folder"]

	BATCH_SIZE = params["model"]["batch_size"]
	LEARNING_RATE = params["model"]["learning_rate"]
	DECAY = params["model"]["decay"]
	LOSS = LOSS_DICT[params["model"]["loss_function"]]

	METRICS = [sparse_categorical_accuracy, precision, recall, iou, true_positive, false_negative, false_positive, true_negative]

	_, _, val_x, val_y, val_us_x, val_us_y = load_sequence_data(SEQUENCE_FOLDER, for_training=False)

	INPUT_SHAPE = val_x.shape[1:]

	model = get_model(INPUT_SHAPE, MODEL_PATH)

	## PREDICT

	predictions = model.predict(val_x,
								batch_size=BATCH_SIZE,
								verbose=1)
	save_predictions((val_us_x, val_us_y), predictions)

	## EVAL

	opt = Adam(learning_rate=LEARNING_RATE, decay=DECAY)

	model.compile(loss=LOSS,
				optimizer=opt,
				metrics=METRICS)

	evaluation = model.evaluate(val_x, val_y,
								batch_size=BATCH_SIZE,
								verbose=1,
								)
	evaluation, matrix = split_results(evaluation)
	save_results(evaluation, [model.loss] + METRICS[:-4])

	## MATRIX

	save_confusion_matrix(matrix)
