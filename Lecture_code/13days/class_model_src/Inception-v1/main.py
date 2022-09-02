from data import TrainDataLoader
from model import InceptionV1
from tensorflow.keras.losses import SparseCategoricalCrossentropy

if __name__ == "__main__":
    ##########################################################
    # NOTE: Training Task
    #
    # NOTE: 1. Load Data
    loader = TrainDataLoader(None)
    (train_images, train_labels) = loader.load_data()
    train_images = train_images / 255.0

    # NOTE: 2. Model(Computation Graph) build (model.py)
    model = InceptionV1(output_dim=2)

    # NOTE: 3-1. traing hyper-parameter (train.py)
    model.compile(
        optimizer="adam",
        # loss=SparseCategoricalCrossentropy(from_logits=True),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # NOTE: 3-2. Run (train.py)
    hist = model.fit(train_images, train_labels, epochs=10)
    model.evalute()
