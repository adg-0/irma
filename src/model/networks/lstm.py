from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization


def lstm_bu(input_shape):

	model = Sequential()

	model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(128, input_shape=input_shape))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(32, activation="relu"))
	model.add(Dropout(0.2))

	model.add(Dense(2, activation="softmax"))

	return model





def lstm(input_shape):

	model = Sequential()

	model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(128, input_shape=input_shape))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(32, activation="relu"))
	model.add(Dropout(0.2))

	model.add(Dense(2, activation="softmax"))

	return model
