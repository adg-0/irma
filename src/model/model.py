from .networks.lstm import lstm


def get_model(input_shape, pretrained_weights=None):

	model = lstm(input_shape)

	if pretrained_weights:
		model.load_weights(pretrained_weights)

	return model
