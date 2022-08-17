from model import InceptionV1
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.losses import SparseCategoricalCrossentropy

if __name__ == "__main__":
    # NOTE: 1. Load (data.py)
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images / 255

    # NOTE: 2. Model(Computation Graph) build (model.py)
    model = InceptionV1(output_dim=10)

    # NOTE: 3-1. traing hyper-parameter (train.py)
    model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # NOTE: 3-2. Run (train.py)
    hist = model.fit(train_images, train_labels, epochs=10)
