import os
import sys
from shutil import copyfile

import numpy as np
import tensorflow as tf
from keras import callbacks
from keras import backend as K


# keras callback -> https://keras.io/callbacks/ ReduceLROnPlateau + Tensorboard
# add learning rate to TB http://seoulai.com/2018/02/06/keras-and-tensorboard.html


class TensorboardCallback(callbacks.TensorBoard):
    def __init__(self, metric_names, log_dir='./log', log_every=1, **kwargs):

        training_log_dir = os.path.join(log_dir, 'training')
        super(TensorboardCallback, self).__init__(log_dir=training_log_dir, **kwargs)

        self.val_log_dir = os.path.join(log_dir, 'validation')
        self.log_every = log_every
        self.counter = 0

        dict_tmp = {}
        for name in ["loss"] + metric_names:
            dict_tmp[name] = 0.
        self.sliding_means_metrics = dict_tmp

        self.val_writer = tf.summary.FileWriter(self.val_log_dir)

    def write_to_tensorboard(self, name, value):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        self.writer.add_summary(summary, self.counter)

    def on_batch_end(self, batch, logs=None):
        self.compute_and_write_metrics_to_tensorboard(batch, logs)

    # learning rate logs
    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super(TensorboardCallback, self).on_epoch_end(epoch, logs)

    def compute_and_write_metrics_to_tensorboard(self, batch, logs):
        for name, _ in logs.items():
            if name in ['batch', 'size']:
                continue
            else:
                self.sliding_means_metrics[name] += logs.get(name)

        if self.counter % self.log_every == 0:
            for name, value in self.sliding_means_metrics.items():
                self.write_to_tensorboard(name, value / (batch + 1))

        self.writer.flush()
        self.counter += 1

    def on_epoch_end(self, epoch, logs=None):
        # super method not called because general behaviour resets the curves. We don't want that
        logs = logs or {}
        self.write_validation_logs_to_tensorboard(logs)
        # self.write_other_logs_to_tensorboard(logs)

        self.reset_sliding_means()

    def write_validation_logs_to_tensorboard(self, logs):
        validation_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in validation_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            # todo ajouter un writer parametrable dans writeTB ??
            self.val_writer.add_summary(summary, self.counter)
        self.val_writer.flush()

    def write_other_logs_to_tensorboard(self, logs):
        other_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        for name, value in other_logs.items():
            if not name in ['batch', 'size',]:
                # self.write_to_tensorboard(name, value / (self.counter + 1))
                self.write_to_tensorboard(name, value)
        self.writer.flush()

    def reset_sliding_means(self):
        for name, _ in self.sliding_means_metrics.items():
            self.sliding_means_metrics[name] = 0.

    def on_train_end(self, logs=None):
        super(TensorboardCallback, self).on_train_end(logs)
        self.val_writer.close()


class BackUpPreviousModel(callbacks.ModelCheckpoint):

    def __init__(self, filepath, **kwargs):
        basepath, ext = os.path.splitext(filepath)
        backup_filepath = basepath + "_backup" + ext

        self.backup_filepath = backup_filepath

        super(BackUpPreviousModel, self).__init__(filepath=filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists(self.filepath):
            copyfile(self.filepath, self.backup_filepath)
        super(BackUpPreviousModel, self).on_epoch_end(epoch, logs)


class CustomEarlyStopping(callbacks.EarlyStopping):

    def __init__(self, save_dir, **kwargs):
        self.filepath = save_dir[:-len(os.path.basename(save_dir))]
        super(CustomEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            basename = "early_stopping_epoch_%05d" % (self.stopped_epoch + 1)
            open(self.filepath + basename, 'a')
        super(CustomEarlyStopping, self).on_train_end(logs)


class CosineAnnealingCallback(callbacks.Callback):
    """
    cf https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/
    """

    def __init__(self, nb_epochs, nb_cycles, lr_max, verbose=0):
        super(CosineAnnealingCallback, self).__init__()
        self.nb_epochs = nb_epochs
        self.nb_cycles = nb_cycles
        self.lr_max = lr_max
        self.verbose = verbose

    def cosine_annealing(self, epoch):
        epochs_per_cycle = np.floor(self.nb_epochs / self.nb_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return (self.lr_max / 2) * (np.cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cosine_annealing(epoch)
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealing updating learning rate to %s.' % (epoch + 1, lr))


# Horovod
class LogTrainingMetricsCallback(callbacks.Callback):
    def __init__(self):
        super(LogTrainingMetricsCallback, self).__init__()
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        metrics_log = self.build_metrics_log(logs)
        # print(self.params)
        sys.stdout.write('{}/{} ... {}\r'.format(batch, self.params['steps'], metrics_log))
        sys.stdout.flush()

    def build_metrics_log(self, logs):
        logs = logs or {}
        self.seen += logs.get('size', 0)
        metrics_log = ''
        for parameter in self.params['metrics']:
            metrics_log = log_batch_metric(logs, metrics_log, parameter)
        return metrics_log


def log_batch_metric(logs, metrics_log, batch_metric_name):
    if batch_metric_name in logs:
        batch_metric_value = logs[batch_metric_name]
        if abs(batch_metric_value) > 1e-3:
            metrics_log += ' - %s: %.4f' % (batch_metric_name, batch_metric_value)
        else:
            metrics_log += ' - %s: %.4e' % (batch_metric_name, batch_metric_value)
    return metrics_log

