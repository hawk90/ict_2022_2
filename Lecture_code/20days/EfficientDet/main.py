from model import Efficient
from tensorflow import keras
from data import Dataset
import tensorflow as tf

N_EPOCHS = 10

if __name__ == "__main__":
    # INFO: 1. Load data
    ds = Dataset(batch_size=32)

    # INFO: 2. Model
    inputs = keras.layer.Input(shape=(224, 224, 3))
    backborn = Efficient()(inputs)
    class_out = keras.layers.Dense(1000)(backborn)
    bbox_out = keras.layers.Dense(4)(backborn)

    model = keras.Model(input=inputs, output=[class_out, bbox_out])

    # INFO: 3. compile and train
    loss_categorical = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    loss_mse = tf.keras.losses.MeanSquaredErro()
    optimizer = tf.keras.optimizers.Adam()
    w1 = 0.8
    w2 = 0.2

    for epoch in range(1, N_EPOCHS + 1):
        print("Epoch {}/{}".format(epoch, N_EPOCHS))
        for step in range(1, (ds.num_train_dataset // ds.batch_size) + 1):
            x_batch, y_batch = ds()
            with tf.GradientTape() as tape:
                class_predict, bbox_prdict = model(x_batch)
                main_loss = loss_categorical(y_batch['class'], class_predict)
                aux_loss = loss_mse(y_batch['bbox'], bbox_prdict)
                loss = w1*main_loss + w2*aux_loss

            gradients = tape.gradient(loss, model.trainable_variables)  # back
            optimizer.apply_gradients(zip(gradients,
                                          model.trainable_variables))
