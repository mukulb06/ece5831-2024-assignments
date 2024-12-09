import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
""" 
Multiclass classification (multiple news categories) with softmax activation and categorical crossentropy loss.
"""
class Reuters:
    def __init__(self, max_words=10000, maxlen=500):
        self.max_words = max_words
        self.maxlen = maxlen
        self.model = None
        self.history = None

    def prepare_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = reuters.load_data(num_words=self.max_words)
        
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.x_train = self.tokenizer.sequences_to_matrix(self.x_train, mode='binary')
        self.x_test = self.tokenizer.sequences_to_matrix(self.x_test, mode='binary')
        
        self.num_classes = np.max(self.y_train) + 1
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_classes)

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.max_words,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=10, batch_size=32):
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

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")