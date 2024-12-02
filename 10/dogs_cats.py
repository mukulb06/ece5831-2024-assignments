import pathlib
import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class DogsCats:
    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('dogs-vs-cats')
    SRC_DIR = pathlib.Path('dogs-vs-cats-original/train')
    EPOCHS = 20

    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def make_dataset_folders(self, subset_name, start_index, end_index):
        for category in self.CLASS_NAMES:
            dir = self.BASE_DIR / subset_name / category
            if not os.path.exists(dir):
                os.makedirs(dir)
            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            for file in files:
                shutil.copyfile(src=self.SRC_DIR / file, dst=dir / file)

    def _make_dataset(self, subset_name):
        return tf.keras.utils.image_dataset_from_directory(
            self.BASE_DIR / subset_name,
            image_size=self.IMAGE_SHAPE[:2],
            batch_size=self.BATCH_SIZE
        )

    def make_dataset(self):
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('validation')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation=True):
        inputs = layers.Input(shape=self.IMAGE_SHAPE)
        x = layers.Rescaling(1./255)(inputs)

        if augmentation:
            x = layers.RandomFlip("horizontal")(x)
            x = layers.RandomRotation(0.1)(x)

        x = layers.Conv2D(32, 3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(128, 3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, model_name):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_name, save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]

        history = self.model.fit(
            self.train_dataset,
            epochs=self.EPOCHS,
            validation_data=self.valid_dataset,
            callbacks=callbacks
        )

        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss')
        plt.show()

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    def predict(self, image_file):
        img = tf.keras.preprocessing.image.load_img(
            image_file, target_size=self.IMAGE_SHAPE[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = predictions[0][0]
        result = "Dog" if score > 0.5 else "Cat"
        confidence = score if score > 0.5 else 1 - score

        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Prediction: {result} (Confidence: {confidence:.2f})")
        plt.show()

        print(f"This image is most likely a {result} with {confidence:.2%} confidence.")