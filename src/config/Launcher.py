from data.data_downloader import download_data
from data.data_sequencer import save_sequence_data
from training.train import train
from prediction.predict import predict


class Launcher(object):

    def __init__(self, action, params):
        self.actions = dict()
        self.params = params
        self.action = action

    def __download_data(self):
        download_data(self.params)

    def __sequence_data(self):
        save_sequence_data(self.params)

    def __train(self):
        train(self.params)

    def __predict(self):
        predict(self.params)

    def __get_launcher(self):
        actions_dict = dict()
        actions_dict["dl_data"] = self.__download_data
        actions_dict["seq_data"] = self.__sequence_data
        actions_dict["train"] = self.__train
        actions_dict["predict"] = self.__predict
        return actions_dict[self.action]

    def launch(self):
        launcher = self.__get_launcher()
        launcher()
