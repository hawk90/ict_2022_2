import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader:
    def __init__(self):
        self.raw, self.info = tfds.load(
            "tf_flowers", as_supervised=True, with_info=True
        )
        self.dataset_size = self.info.splits["train"].num_examples
        self.n_classes = self.info.features["label"].num_classes
        self.train = list(map(self.preprocess, self.raw))

    def __str__(self):
        return f"#Image: {self.dataset_size}, #Classes: {self.n_classes}"

    def preprocess(self, feature_dict):
        # image = tf.image.resize(feature_dict["image"], [224, 224])
        print(feature_dict["image"])
        return feature_dict["image"], feature_dict["label"]
