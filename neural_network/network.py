import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow


def ReLU(i):
    return np.maximum(0, i)


def relu_prime(x):
    return (x > 0).astype(float)


def MSE(output, expected_output):
    return np.sum(
        np.square(np.subtract(expected_output, output)))/len(expected_output)


class Layer(object):
    def __init__(self, input_size, output_size, name="Layer"):
        self.name = name
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size, 1)


class Network(object):
    def __init__(self, input_layer_size):
        self.input_layer_size = input_layer_size
        self.layers = []

    def add_layer(self, layer_size):
        """Adds a new layer to the network."""
        self.layers.append(Layer(
            input_size=self.layers[-1].weights.shape[0] if self.layers else self.input_layer_size,
            output_size=layer_size,
        ))
        print(self.layers[-1].weights.shape)

    def forward(self, input_data):
        """
        Performs a forward pass through the neural network.

        Args:
            input_data (numpy.ndarray): Input data to be passed through the network.

        Returns:
            numpy.ndarray: The output of the network after applying each layer's weights, biases, and the ReLU activation function.
        """
        for layer in self.layers:
            input_data = ReLU(np.dot(layer.weights, input_data) + layer.biases)
        return input_data

    def SGD(self, features, labels,  epochs,
            batch_size, learning_rate, validation_split=None):
        # zip features and labels together
        training_data = list(zip(features, labels))
        # If validation_split is provided, split the training data
        if validation_split:
            split_index = round(
                int(len(training_data) * (1 - validation_split)))
            training_data, validation_data = training_data[:
                                                           split_index], training_data[split_index:]
        # Perform stochastic gradient descent
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i:i + batch_size]
                self.z<(batch_data, learning_rate)
            if validation_split:
                print(
                    f'Epoch: {epoch}: {self.evaluate(validation_data)/len(validation_data)}')

    def update_batch(self, batch_data, learning_rate):
        nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]

        for feature, label in batch_data:
            delta_nabla_b, delta_nabla_w = self.back_propagate(feature, label)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update weights and biases
        for i in range(len(self.layers)):
            self.layers[i].biases = self.layers[i].biases - \
                (learning_rate / len(batch_data)) * nabla_b[i]
            self.layers[i].weights = self.layers[i].weights - \
                (learning_rate / len(batch_data)) * nabla_w[i]

    def back_propagate(self, feature, label):
        nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]

        # Forward pass
        activation = feature
        # List to store all activations, layer by layer
        activations = [activation]
        zs = []  # List to store all z vectors, layer by layer

        for layer in self.layers:
            z = np.dot(layer.weights, activation) + layer.biases
            zs.append(z)
            activation = ReLU(z)
            activations.append(activation)

        # backward pass
        delta = (activations[-1] - label) * \
            relu_prime(zs[-1])  # Assuming MSE loss
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagate the error
        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.layers[-l+1].weights.transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


net = Network(input_layer_size=11)

net.add_layer(3)
net.add_layer(5)
net.add_layer(3)

# sizes = [11,3,5,3]
# biases = [np.random.randn(y, 1) for y in sizes[1:]]
# weights = [np.random.randn(y, x)
#                 for x, y in zip(sizes[:-1], sizes[1:])]
# for index, layer in enumerate(net.layers):
#     print(f'3bbais: {biases[index].shape}')
#     print(f'biases: {layer.biases.shape}')
#     print(f'3bweights: {weights[index].shape}')
#     print(f'weights: {layer.weights.shape}')


df = pd.read_csv(
    '/home/eshulman/git/AI/TensorFlow/datasets/Multiclass_Diabetes_Dataset.csv')

# Convert to numpy array
dataset_np = df.to_numpy()

# Split dataset into features and diagnosis
features = dataset_np[:, :-1]
diagnostics = dataset_np[:, -1]

# Scale features to unify the scale
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert diagnostics to one-hot encoding
diagnostics = tensorflow.keras.utils.to_categorical(diagnostics)

# Split the dataset into training and testing sets
features_train, features_test, diagnostics_train, diagnostics_test = train_test_split(
    features, diagnostics, test_size=0.2)
print(features_train)
net.SGD(features_train, diagnostics_train, epochs=10,
        batch_size=32, learning_rate=0.01, validation_split=0.2)
