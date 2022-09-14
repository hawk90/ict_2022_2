from data import Dataloader
import tensorflow as tf
import model
from utils import DisplayCallback

BATCH_SIZE = 32
BUFFER_SIZE = 1000
EPOCHS = 20


if __name__ == "__main__":
    # NOTE: 1. Data load
    dl = Dataloader(download=True)
    NUM_TRAIN = dl.num_train
    STEPS_PER_EPOCH = NUM_TRAIN // BATCH_SIZE

    train_dataset = dl.train
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(1)

    # NOTE: 2. Model build
    seg_model = model.unet_model()
    seg_model.summary()

    # NOTE: 3. Compile and Train
    seg_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    model_history = seg_model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=[DisplayCallback()],
    )
