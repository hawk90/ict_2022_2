from tensorflow.keras import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten

# 01_Data Load
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 02_Model V1
model = Sequential(
    [
        Flatten(input_shape=[28, 28]),
        Dense(300, activation="relu"),
        Dense(100, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)


# 03_Model Summary
model.summary()

# 04_Handling Model
print(model.layers)

hidden1 = model.layers[1]
print(hidden1.name)

print(model.get_layer("dense") is hidden1)

weights, biases = hidden1.get_weights()
print(f"Weights: {weights}")
print(f"Shape of Weights: {weights.shape}")
print(f"Biases: {biases}")
print(f"Shape of Biases: {biases.shape}")

# 05_Model Compile
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
)

# 06_Model fit
history = model.fit(x_train, y_train, epochs=30)
