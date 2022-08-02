import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import SGD

class WideAndDeepModel(Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(units, activation=activation)
        self.hidden2 = Dense(units, activation=activation)
        self.main_output = Dense(1)
        self.aux_output = Dense(1)

    def call(self, inputs):
        input_a, input_b = inputs
        hidden1 = self.hidden1(input_b)
        hidden2 = self.hidden2(hidden1)
        concat = Concatenate()([input_a, input_b])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

# 01_Data Load
housing = fetch_california_housing()

x_train_full, x_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target)
x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)


x_train_a, x_train_b = x_train[:, :5], x_train[:, 2:]
x_valid_a, x_valid_b = x_valid[:, :5], x_valid[:, 2:]
x_test_a, x_test_b = x_test[:, :5], x_test[:, 2:]

# 02_Model
model = WideAndDeepModel()


# 03_Model Compile
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1],
        optimizer=SGD(lr=1e-3))

# 04_Model fit
history = model.fit([x_train_a, x_train_b], [y_train, y_train], epochs=20,
        validation_data=([x_valid_a, x_valid_b], [y_valid, y_valid]))

mse_test = model.evaluate([x_test_a, x_test_b], [y_test, y_test])
