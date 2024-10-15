import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Function to load labels from labels.txt
def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Function to load the model
def load_my_model(model_file):
    return tf.keras.models.load_model(model_file)

# Function to preprocess the image for prediction
def preprocess_image(image_path, img_size=(224, 224)):
    image = cv2.imread(image_path)
    img = cv2.resize(image, img_size)  # Resize to match the model's input size
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return image, img

# Function to predict the class of the image
def predict_image(model, img, class_names):
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence_score = predictions[0][class_idx]
    prediction_label = class_names[class_idx]
    return prediction_label, confidence_score

# Function to display the image with prediction
def display_prediction(image, label, confidence_score):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {label} (Confidence: {confidence_score:.4f})')
    plt.show()

# Function to print the prediction result
def print_prediction(label, confidence_score):
    print(f'Class: {label}')
    print(f'Confidence Score: {confidence_score:.4f}')

# Main function to orchestrate the prediction pipeline
def main():
    model_file = 'keras_model.h5'
    label_file = 'labels.txt'
    image_path = "paper1.jpg"  # You can replace this with a command-line argument or other image source

    # Load model and labels
    model = load_my_model(model_file)
    class_names = load_labels(label_file)

    # Preprocess the image
    original_image, preprocessed_image = preprocess_image(image_path)

    # Predict the class and confidence score
    label, confidence_score = predict_image(model, preprocessed_image, class_names)

    # Print and display the prediction
    print_prediction(label, confidence_score)
    display_prediction(original_image, label, confidence_score)

# Execute the main function
if __name__ == '__main__':
    main()
