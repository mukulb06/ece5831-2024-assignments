import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
import numpy as np
""" 
 Regression (predicting continuous house prices) with no activation in the final layer and mean squared error loss.
"""
class BostonHousing:
    def __init__(self):
        self.model = None
        self.history = None

    def prepare_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
        
        mean = self.x_train.mean(axis=0)
        std = self.x_train.std(axis=0)
        self.x_train = (self.x_train - mean) / std
        self.x_test = (self.x_test - mean) / std

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, epochs=100, batch_size=32):
        self.history = self.model.fit(self.x_train, self.y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=0.2,
                                      verbose=1)

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()


    def evaluate(self):
        loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Loss (MSE): {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")