import os

from config.Launcher import Launcher
from config.argparser import argparser

if __name__ == '__main__':

    action, params = argparser()

    launcher = Launcher(action=action, params=params)
    launcher.launch()
