from model import VGG
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images / 255

    model = VGG()

    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    hist = model.fit(train_images, train_labels, epochs=10)
