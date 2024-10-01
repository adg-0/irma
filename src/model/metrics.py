from keras import backend as K


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# tp / tp + fp + fn
def iou(y_true, y_pred):
    intersection = K.sum(y_pred * y_true)
    union = K.sum(y_pred) + K.sum(y_true) - intersection + K.epsilon()
    return K.mean(intersection / union)


# harmonic mean of precision and recall
def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


# weighs recall higher than precision (by placing more emphasis on false negatives)
def f2(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return (5 * prec * rec) / (4 * prec + rec + K.epsilon())


def true_positive(y_true, y_pred):
    return K.round(K.clip(y_true * y_pred, 0, 1))


def false_positive(y_true, y_pred):
    return K.round(K.clip((1 - y_true) * y_pred, 0, 1))


def true_negative(y_true, y_pred):
    return K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1))


def false_negative(y_true, y_pred):
    return K.round(K.clip(y_true * (1 - y_pred), 0, 1))
