import sys
import argparse
import matplotlib.pyplot as plt
from mnist_data import MnistData

''' 
To run the program, write in the command module
python module5-2.py [train/test] [number]
for example, 
python module5-2.py train 109
'''

def main():
    # Parsing the arguments
    parser = argparse.ArgumentParser(description="Test MnistData class")
    parser.add_argument('dataset_type', choices=['train', 'test'], help="Specify 'train' or 'test' dataset")
    parser.add_argument('index', type=int, help="Specify the index of the image")
    
    args = parser.parse_args()
    
    # Load the dataset
    mnist_data = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load()
    
    # Select dataset based on the argument
    if args.dataset_type == 'train':
        images = train_images
        labels = train_labels
    else:
        images = test_images
        labels = test_labels

    # Get the image and the label at the specified index
    image = images[args.index].reshape(28, 28)  # Reshape the flattened image to 28x28
    label = mnist_data.get_original_labels(labels[args.index])

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

    # Print the label
    print(f'The label for the image at index {args.index} is: {label}')

if __name__ == "__main__":
    main()

