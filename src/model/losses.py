from keras import backend as K
import tensorflow as tf


def IoU(y_true, y_pred):
    """
    Intersection over Union
    Soft-Jaccard loss
    """
    pred_flat = K.flatten(y_pred)
    true_flat = K.flatten(y_true)
    intersection = K.sum(pred_flat * true_flat) + 1
    denominator = K.sum(pred_flat) + K.sum(true_flat) - intersection + 1
    return (1 - K.mean(intersection / denominator)) * 100


# binary
def weighted_cross_entropy(p=0.5):
    def loss(y_true, y_pred):
        pred_flat = K.flatten(y_pred)
        true_flat = K.flatten(y_true)
        val1 = - (1.0 - p) * K.log(pred_flat + K.epsilon()) * true_flat
        val2 = - p * K.log(1.0 - pred_flat + K.epsilon()) * (1.0 - true_flat)
        return K.mean(val1 + val2)
    return loss


# binary
def IoU_cross_entropy(p=0.5):
    """
    A combination of soft-Jaccard loss and Weighted-CrossEntropy
    """
    def loss(y_true, y_pred):
        pred_flat = K.flatten(y_pred)
        true_flat = K.flatten(y_true)
        val1 = -(1 - p) * K.log(pred_flat + K.epsilon()) * true_flat
        val2 = -p * K.log(1.0 - pred_flat + K.epsilon()) * (1.0 - true_flat)
        intersection = 2 * K.sum(pred_flat * true_flat) + 1
        denominator = K.sum(pred_flat) + K.sum(true_flat) + 1
        return -K.log(K.mean(intersection / denominator) + K.epsilon()) + K.mean(val1 + val2)
    return loss
