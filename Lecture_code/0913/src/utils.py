import matplotlib.pyplot as plt
import tensorflow as tf

from IPython.display import clear_output

TITLE = ["input", "true", "predicted"]


def display(examples):
    num_examples = len(examples)
    plt.figure(figsize=(15, 15))

    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        plt.title(TITLE[i])
        plt.imshow(tf.keras.utils.array_to_img(examples[i]))
        plt.axis("off")
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dim(pred_mask, axis=-1)
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for img, mask in dataset.take(num):
            pred_mask = model.predict(img)
            display(img[0], mask[0], create_mask(pred_mask))


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()


def display_losses(losses, epochs):
    loss, val_loss = losses

    plt.figure()
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.plot(epochs, loss, "r")
    plt.plot(epochs, val_loss, "g")

    plt.show()
