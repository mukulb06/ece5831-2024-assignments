from mnist import Mnist
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp 
import numpy as np   
import matplotlib.pyplot as plt
import pickle 

# Initialize MNIST data and the two-layer network
mnist = Mnist()
(x_train, y_train), (x_test, y_test) = mnist.load()
network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=500, output_size=10)

iterations = 10000
train_size = x_train.shape[0]
batch_size = 16
lr = 0.01

iter_per_epoch = max(train_size // batch_size, 1)

train_losses = []
train_accs = []
test_accs = []

for i in range(iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)

    # Update parameters and store in uppercase keys to match mnist.py expectations
    for key in ('w1', 'b1', 'w2', 'b2' ):
        network.params[key] -= lr * grads[key.lower()]

    # Record training loss
    train_losses.append(network.loss(x_batch, y_batch))

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f'Epoch {i // iter_per_epoch}: train acc = {train_acc}, test acc = {test_acc}')

# Plot accuracy over epochs
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_accs))
plt.plot(x, train_accs, label='train acc')
plt.plot(x, test_accs, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# Save the trained parameters to pickle file with uppercase keys
my_weight_pkl_file = 'bhatia_mnist_model.pkl'
with open(my_weight_pkl_file, 'wb') as f:
    print(f'Pickle: {my_weight_pkl_file} is being created.')
    pickle.dump(network.params, f)
    print('Done.')
