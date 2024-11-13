# ece5831-2024-assignments

# 02

(2) learn-python-2.ipynb is the file for Data Science Tutorials from https://www.learnpython.org/

(3) learn-python-3.ipynb is Advanced Tutorials in the respective order from https://www.learnpython.org/

(4) python-tutorial-1-10.ipynb is Section 1-10 from https://www.pythontutorial.net/. with all parts of the sections in the respective order

(5) python-tutorial-11.py is Section 11 from https://www.pythontutorial.net/ with all the dependecies

(6) python-tutorial-12-15.ipynb is Section 12,13 and 15 from https://www.pythontutorial.net/ with all parts of the sections in respective order

(7) numpy-tutorials.ipynb is the Numpy tutorial from https://cs231n.github.io/python-numpy-tutorial/

billing.py, pricing.py and product.py are dependencies for (5) python-tutorial-11.py

cars.csv, hello.txt and test.txt are dependencies for Section 12 in (6) python-tutorial-12-15.ipynb

sales in a depository for Section 13 in (6) python-tutorial-12-15.ipynb

# 03

logic_gate.py is a python file, with a class LogicGate and functions for all logic gates inside 

module3.py is where i implemented all the logic gates with test cases

module3.ipynb is detailed with every heading for every part of the code

# 04

multilayer_perceptron.py is the file with class MultilayerPerceptron that uses functions to solve for three layers of perceptron.

module4.py tests and validates the script from multilayer_perceptron.py

module4.ipynb is a detailed and explained version of first trying step function, then sigmoid function and plotting their graphs, to multilayer perceptron 
 
# 05

Video Link for rock-paper-scissor-live.py => https://youtu.be/0vIVxSZjS68 

rock_paper_scissors.py uses the model keras_model.h5 to get predictions for a single iamge of class paper with a confidence score of 1.000

rock_paper_scissors_live.py uses the same model to predict rock, paper, or scissor using webcam 

teachables.ipynb is using rock_paper_scissors.py to test the model in different images (2 images of each class)

# 05-2

module5-2.ipynb shows the implementation of different functions, softmax function, updates softmax function, _download(), _download_all(), _load_images(), _load_labels(), _create_dataset(), _change_one_hot_label(), _init_dataset,   and MnistData class

mnist_data.py is for MnistData class, when the script is run it prints MnistData class is to load MNIST datasets, and   Returns(train_images, train_labels), (test_images, test_labels) 

module5-2.py tests mnist_data.py, to run module5-.py write in the command module
python module5-2.py [train/test] [number]
for example, 
python module5-2.py train 109
and it shows the image with the label, and the label is also printed in terminal. 


# 05-3
mnist.py is Mnist class 

module5_3.py uses the Mnist class and the mnist data set to predict handwritten images, the format to run the file for a specific image is 
python module5_3.py "./Custom MNIST Samples/Digit 0/0_3.png" 0
python module5_3.py "image path" expected digit

module5_3.ipynb tests the functions of Mnist class and tests module5_3.py on every single image file that we created using the command 
!python module5_3.py "./Custom MNIST Samples/Digit 0/0_3.png" 0
!python module5_3.py "image path" expected digit


# 06 
errors.py has the Errors class
layers.py has Relu, Sigmoid, Affine, SoftmaxWithLoss Classes
module6.py uses the model made by train.py to test the mnist dataset
two_layer_net_with_back_prop has the TwoLayerNetWithBackProp Class
activations.py has the Activation class
