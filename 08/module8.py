import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from le_net import LeNet  # Importing the LeNet class

def preprocess_image(image_path):
    """Preprocess an image for prediction."""
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28, 28))
    # Normalize the image and reshape to match model input
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)  # Batch size 1, channel 1
    return image

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognition using LeNet.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("true_digit", type=int, help="Actual digit in the image.")
    args = parser.parse_args()

    # Define the model name
    model_name = "bhatia_cnn_model.keras"

    # Create an instance of the LeNet class
    lenet = LeNet(batch_size=32, epochs=20)

    # Check if the model already exists
    if os.path.exists(model_name):
        print(f"Model {model_name} found. Loading the saved model...")
        lenet.load("bhatia_cnn_model")  # Load the existing model
    else:
        print("No saved model found. Training a new model...")
        # Train the model
        lenet.train()
        # Save the model
        lenet.save("bhatia_cnn_model")
        print(f"Model saved as {model_name}")

    # Predict the digit from the image
    try:
        print(f"Processing image: {args.image_path}")
        image = preprocess_image(args.image_path)
        prediction = lenet.predict(image)
        predicted_digit = prediction[0]

        # Display the image
        img_to_display = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_to_display, cmap='gray')
        plt.title(f"Predicted Digit: {predicted_digit}")
        plt.axis('off')
        plt.show()

        # Check the prediction
        if predicted_digit == args.true_digit:
            print(f"Success: Image {args.image_path} is for digit {args.true_digit} and recognized as {predicted_digit}.")
        else:
            print(f"Fail: Image {args.image_path} is for digit {args.true_digit} but the inference result is {predicted_digit}.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
