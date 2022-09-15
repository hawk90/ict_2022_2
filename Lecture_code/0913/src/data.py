import tensorflow_datasets as tfds
import tensorflow as tf


RESIZE = (128, 128)


class Dataloader:
    def __init__(self, download=False):
        self.dataset, self.info = tfds.load(
            "oxford_iiit_pet", data_dir="../data", download=download, with_info=True
        )

    def _normalize(self, img, mask):
        img = tf.cast(img, tf.float32) / 255.0
        mask = mask - 1  # {1, 2, 3} -> {0, 1, 2}
        return img, mask

    def _load_train(self, data):
        img = tf.image.resize(data["image"], RESIZE)
        mask = tf.image.resize(data["segmentation_mask"], RESIZE)
        return self._normalize(img, mask)

    def _load_test(self, data):
        img = tf.image.resize(data["image"], RESIZE)
        mask = tf.image.resize(data["segmentation_mask"], RESIZE)
        return self._normalize(img, mask)

    @property
    def train(self):
        return self.dataset["train"].map(self._load_train)

    @property
    def test(self):
        return self.dataset["test"].map(self._load_test)

    @property
    def num_train(self):
        return self.info.splits["train"].num_examples
