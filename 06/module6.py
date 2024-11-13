import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mnist
from mnist_data import MnistData 

""" 
the format to run the file for a specific image is 
python module6.py "./Custom MNIST Samples/Digit 0/0_3.png" 0
python module6.py "image path" expected digit

"""

mnist = mnist.Mnist()
mnist.init_network()

def load_and_predict_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image from path {image_path}")
        return None

    # Resize the image to 28x28 if it isn't already
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
    
    # Normalize and flatten the image
    img = img.astype(np.float32)
    img /= 255.0
    img = img.reshape(-1)

    # Predict using the initialized Mnist network
    prediction = mnist.predict(img)
    predicted_label = np.argmax(prediction)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show() 

    return predicted_label

def main():
    parser = argparse.ArgumentParser(description="Predict MNIST digit from a custom image and verify the result.")
    parser.add_argument("image_filename", type=str, help="The filename of the image to predict")
    parser.add_argument("expected_digit", type=int, help="The expected digit for the image")
    args = parser.parse_args()

    # Predict the label for the input image
    predicted_label = load_and_predict_image(args.image_filename)
    
    if predicted_label is None:
        print("Prediction failed due to an image loading error.")
        return

    # Check if the prediction matches the expected digit
    if predicted_label == args.expected_digit:
        print(f"Success: Image {args.image_filename} is for digit {args.expected_digit} and is recognized as {predicted_label}.")
    else:
        print(f"Fail: Image {args.image_filename} is for digit {args.expected_digit} but the inference result is {predicted_label}.")

if __name__ == "__main__":
    main()
