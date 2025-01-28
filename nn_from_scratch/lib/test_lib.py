import numpy as np
import urllib.request
import gzip
import os

# URLs for MNIST dataset
MNIST_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
}

def download_and_load_mnist():
    data_dir = "mnist_data"
    os.makedirs(data_dir, exist_ok=True)
    mnist_data = {}
    for key, url in MNIST_URLS.items():
        file_path = os.path.join(data_dir, url.split("/")[-1])
        if not os.path.exists(file_path):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, file_path)
        with gzip.open(file_path, 'rb') as f:
            if "images" in key:
                mnist_data[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
            else:
                mnist_data[key] = np.frombuffer(f.read(), np.uint8, offset=8)
    return mnist_data

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def compute_loss(y_pred, y_true):
    m = y_true.shape[1]
    log_likelihood = -np.log(y_pred[y_true, np.arange(m)])
    return np.sum(log_likelihood) / m

def initialize_parameters(layers_dims):
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters[f"W{l}"] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))
    return parameters

def forward_propagation(X, parameters):
    caches = {}
    A = X.T
    L = len(parameters) // 2
    for l in range(1, L + 1):
        W, b = parameters[f"W{l}"], parameters[f"b{l}"]
        Z = np.dot(W, A) + b
        if l == L:
            A = softmax(Z)
        else:
            A = relu(Z)
        caches[f"A{l-1}"], caches[f"Z{l}"] = A, Z
    return A, caches

def backward_propagation(AL, y_true, parameters, caches):
    grads = {}
    L = len(parameters) // 2
    m = y_true.shape[1]
    Y = np.zeros((AL.shape[0], m))
    Y[y_true, np.arange(m)] = 1
    dZ = AL - Y
    for l in reversed(range(1, L + 1)):
        A_prev = caches[f"A{l-1}"] if l > 1 else X_train.T
        W = parameters[f"W{l}"]
        grads[f"dW{l}"] = np.dot(dZ, A_prev.T) / m
        grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
        if l > 1:
            dA = np.dot(W.T, dZ)
            dZ = dA * relu_derivative(caches[f"Z{l-1}"])
    return grads

def update_parameters(parameters, grads, learning_rate):
    for l in range(1, len(parameters) // 2 + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters

def train_nn(X_train, y_train, layers_dims, epochs, learning_rate):
    parameters = initialize_parameters(layers_dims)
    for epoch in range(epochs):
        AL, caches = forward_propagation(X_train, parameters)
        cost = compute_loss(AL, y_train)
        grads = backward_propagation(AL, y_train, parameters, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")
    return parameters

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    return np.argmax(AL, axis=0)

if __name__ == "__main__":
    # Load MNIST dataset
    mnist = download_and_load_mnist()
    X_train, y_train = mnist["train_images"] / 255.0, mnist["train_labels"]
    X_test, y_test = mnist["test_images"] / 255.0, mnist["test_labels"]

    # Use a subset of data for quick training
    X_train, y_train = X_train[:10000], y_train[:10000]

    # Network configuration
    layers_dims = [784, 128, 64, 10]  # Input, hidden layers, output
    epochs = 100
    learning_rate = 0.01

    # Train the network
    print("Training the network...")
    parameters = train_nn(X_train, y_train, layers_dims, epochs, learning_rate)

    # Evaluate on test set
    print("Evaluating the model...")
    y_pred = predict(X_test, parameters)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


