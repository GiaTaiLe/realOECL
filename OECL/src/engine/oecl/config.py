"""
Author: Le Gia Tai
"""
import os
from pathlib import Path

import yaml
from easydict import EasyDict

from util.utility import mkdir_if_missing


class PathConfig(object):
    """
    Create necessary paths.
    """

    def __init__(self, config_file_env, config_file_exp, times, id_class=None):
        self.config_file_env = config_file_env
        self.config_file_exp = config_file_exp
        self.times = times
        self.id_class = id_class

    @staticmethod
    def database_root(dataset_name=""):
        """
        @param dataset_name:
        @return:
        """
        database_names = {"imagenet30", "imagenet1k"}
        assert dataset_name in database_names

        dataset_store_dir = os.path.join(Path.home(), "Documents/path/to", dataset_name)

        return dataset_store_dir

    def create_config(self):
        """
        :return: paths
        """
        # read config file for file environment
        with open(self.config_file_env, "r") as stream:
            root_dir = yaml.safe_load(stream)["root_dir"]  # where to save files and results
            root_dir = os.path.join(Path.home(), root_dir)

        with open(self.config_file_exp, "r") as stream:
            config = yaml.safe_load(stream)

        cfg = EasyDict()  # easydict dictionary

        # copy config to easydict for better interaction
        for k, v in config.items():
            cfg[k] = v

        if self.id_class is not None:
            cfg["class"] = self.id_class

        # this tf to help create path and ckpt quicker
        def create_sub_ckpt(*args, method_, sub_dataset=None):
            """
            @param args:
            @param method_:
            @param sub_dataset:
            @return:
            """
            # directory
            dir_ = os.path.join(*args)
            mkdir_if_missing(dir_)

            # path
            method_ = str(method_)
            if sub_dataset is not None:
                cfg["{}_ckpt".format(method_)] = os.path.join(dir_, "{}_c{}_t{}_ckpt.pth.tar".
                                                              format(method_, sub_dataset, self.times))
                cfg["{}_model".format(method_)] = os.path.join(dir_, "{}_c{}_t{}_model.pth.tar".
                                                               format(method_, sub_dataset, self.times))
                cfg["{}_setup".format(method_)] = os.path.join(dir_, "{}_c{}_t{}.yaml".
                                                               format(method_, sub_dataset, self.times))

            else:
                cfg["{}_ckpt".format(method_)] = os.path.join(dir_, "{}_t{}_ckpt.pth.tar".
                                                              format(method_, self.times))
                cfg["{}_model".format(method_)] = os.path.join(dir_, "{}_t{}_model.pth.tar".
                                                               format(method_, self.times))
                cfg["{}_setup".format(method_)] = os.path.join(dir_, "{}_t{}.yaml".
                                                               format(method_, self.times))

        base_dir = os.path.join(root_dir, cfg["dataset"])
        mkdir_if_missing(base_dir)

        cfg["base_dir"] = base_dir

        _path = (base_dir, cfg["method"])

        if "class" in cfg.keys():
            create_sub_ckpt(*_path, method_=cfg["method"], sub_dataset=cfg["class"])

        else:
            create_sub_ckpt(*_path, method_=cfg["method"])

        return cfg
