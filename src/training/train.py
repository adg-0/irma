from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.metrics import binary_accuracy, sparse_categorical_accuracy, binary_crossentropy, sparse_categorical_crossentropy

from data.data_sequencer import load_sequence_data
from model.model import get_model
from model.metrics import *


LOSS_DICT = {
	"sparse_categorical_crossentropy": sparse_categorical_crossentropy
}


def train(params):

	SEQUENCE_FOLDER = params["path"]["sequence_folder"]
	LOGS_FOLDER = params["path"]["logs_folder"]
	MODEL_PATH = f"{params['path']['model_folder']}{params['model']['id_name']}.h5"

	SPLIT_RATIO = params["data"]["split_ratio"]

	EPOCHS = params["model"]["epochs"]
	BATCH_SIZE = params["model"]["batch_size"]
	LEARNING_RATE = params["model"]["learning_rate"]
	DECAY = params["model"]["decay"]
	LOSS = LOSS_DICT[params["model"]["loss_function"]]
	# TODO export in config + metrics dict
	METRICS = [sparse_categorical_accuracy, precision, iou]

	train_x, train_y, val_x, val_y, _, _ = load_sequence_data(SEQUENCE_FOLDER, SPLIT_RATIO)

	INPUT_SHAPE = train_x.shape[1:]

	model = get_model(INPUT_SHAPE)

	opt = Adam(learning_rate=LEARNING_RATE, decay=DECAY)

	model.compile(loss=LOSS,
				optimizer=opt,
				metrics=METRICS)

	# TODO make a proper save method, also with custom checkpoint
	tensorboard = TensorBoard(log_dir=LOGS_FOLDER)
	checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

	model.fit(train_x, train_y,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			verbose=1,
			shuffle=True,
			validation_data=(val_x, val_y),
			validation_freq=1,
			callbacks=[tensorboard, checkpoint])
