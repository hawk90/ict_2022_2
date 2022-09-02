import abc
import os
from glob import glob

import numpy as np
from PIL import Image

CONF = {
    "DIR": {"TRAIN": "./data/train", "TEST": "./data/test"},
    "DATA_OPTS": {"VALID": True, "RATE": 0.3},
}
CLASS = ["dog", "cat"]


class DataLoader(metaclass=abc.ABCMeta):
    def __init__(self, conf):
        pass

    @abc.abstractmethod
    def load_data(self):
        pass


class TrainDataLoader(DataLoader):
    def __init__(self, conf):
        pass

    def load_data(self):
        if CONF["DATA_OPTS"]["VALID"]:
            images = []
            labels = []
            for img_path in glob("./data/train/*.jpg"):
                with Image.open(img_path) as img:
                    image = np.array(img)
                label = self.label(img_path)

                images.append(image)
                labels.append(label)

            # TODO: RATE를 이용하여 나누세요
            train_images = images[:100]
            train_labels = images[:100]
            valid_labels = images[100:]
            valid_images = images[100:]

            return (train_images, train_labels), (valid_images, valid_labels)
        else:
            images = []
            labels = []
            for img_path in glob("./data/train/*.jpg"):
                with Image.open(img_path) as img:
                    image = np.array(img)
                label = self.label(img_path)

                images.append(image)
                labels.append(label)

            return (images, labels)

    def label(self, img_path):
        fname = os.path.basename(img_path).lower()
        if fname.startswith("dog"):
            return 0
        return 1


class TestDataLoader(DataLoader):
    #TODO: load_data를 완성 시키세요
    def load_data(self):
        pass
