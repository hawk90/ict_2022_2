import tensorflow_datasets as tfds
import tensorflow as tf


RESIZE = (128, 128)


class Dataloader:
    def __init__(self, download=False):
        self.dataset, self.info = tfds.load(
            "celeb_a", data_dir="../data", download=download, with_info=True
        )

    def _normalize(self, img):
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _load_train(self, data):
        img = tf.image.resize(data["image"], RESIZE)
        return self._normalize(img)

    @property
    def train(self):
        return self.dataset["train"].map(self._load_train)

    @property
    def num_train(self):
        return self.info.splits["train"].num_examples
