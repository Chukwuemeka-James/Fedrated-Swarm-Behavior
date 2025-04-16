import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()

        # Add an Input layer explicitly instead of using input_shape in Dense
        model.add(tf.keras.Input(shape=(shape,)))

        # First hidden layer with 500 units and ReLU activation
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Second hidden layer with 300 units and ReLU activation
        model.add(Dense(300))
        model.add(Activation("relu"))

        # Third hidden layer with 200 units and ReLU activation
        model.add(Dense(200))
        model.add(Activation("relu"))

        # Output layer with number of class units and sigmoid activation
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        return model
