from keras.activations import relu, elu


def leaky_relu(alpha=0.2):
    def leaky(x):
        return relu(x, alpha)
    return leaky


def eLU(alpha=1.0):
    def elu2(x):
        return elu(x, alpha)
    return elu2
