import numpy as np

class MLP:
    def __init__(self, HL=2, epoch=500, lr=0.1):
        self.HL = HL
        self.epoch = epoch
        self.lr = lr

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def d_activation(self, x):
        return x * (1 - x)

    def initialize(self, input_size):
        self.weights = []
        self.biases = []

        # input → first hidden
        self.weights.append(np.random.randn(input_size, input_size))
        self.biases.append(np.zeros((1, input_size)))

        # hidden layers
        for _ in range(self.HL - 1):
            self.weights.append(np.random.randn(input_size, input_size))
            self.biases.append(np.zeros((1, input_size)))

        # last → output
        self.weights.append(np.random.randn(input_size, 1))
        self.biases.append(np.zeros((1, 1)))

    def forwardpath(self, x):
        self.activations = [x]

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.activation(z)
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, y):
        error = self.activations[-1] - y
        deltas = [error * self.d_activation(self.activations[-1])]

        for i in reversed(range(len(self.weights)-1)):
            delta = np.dot(deltas[-1], self.weights[i+1].T) * \
                    self.d_activation(self.activations[i+1])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * np.dot(self.activations[i].T, deltas[i])
            self.biases[i] -= self.lr * deltas[i]

    def fit(self, X, y):
        self.initialize(X.shape[1])

        for epoch in range(self.epoch):
            for i in range(len(X)):
                x = X[i].reshape(1, -1)
                target = y[i].reshape(1, -1)

                self.forwardpath(x)
                self.backward(target)

            if epoch % 100 == 0:
                loss = np.mean((target - self.activations[-1])**2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        outputs = []
        for i in range(len(X)):
            out = self.forwardpath(X[i].reshape(1, -1))
            outputs.append(out[0][0])
        return np.array(outputs)