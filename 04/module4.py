# %% Importing the class from the file multilayer_perceptron.py
from multilayer_perceptron import MultiLayerPerceptron
import numpy as np
import matplotlib.pyplot as plt

mlp = MultiLayerPerceptron()

# Initialize the network
mlp.init_network()

# %%Showing plot for Sigmoid and Step function
x_values = np.linspace(-10, 10, 100)  # Arbitrary range of x

# Apply sigmoid to all x_values
sigmoid_outputs = [mlp.sigmoid(x) for x in x_values]

# Apply step function to all x_values
step_outputs = [mlp.step(x) for x in x_values]

# Plotting the sigmoid function
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_values, sigmoid_outputs, label='Sigmoid')
plt.title("Sigmoid")

# Plotting the step function
plt.subplot(1, 2, 2)
plt.plot(x_values, step_outputs, label='Step')
plt.title("Step")

# Show plots
plt.tight_layout()
plt.show()

# %% Using multilevel_perceptron.py to calculate sigmoid, step and forward

# Sample input
input_arr = np.array([0.2, 0.8])

# Use the sigmoid function
sigmoid_output = mlp.sigmoid(input_arr)
print("Sigmoid output:", sigmoid_output)

# Use the step function
step_arr = np.array([1, 200, -2, 0, 0.5])
step_output = mlp.step(step_arr)
print("Step output:", step_output)

# Use the forward function
forward_output = mlp.forward(input_arr)
print("Forward output:", forward_output)
