import numpy as np
import matplotlib.pyplot as plt

def create_toy_dataset(n_samples=100):
    x = np.linspace(-1, 1, n_samples)
    y = 0.1 * x + np.power(x, 2) + np.power(x, 3)
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    return x, y

def mse(y_true, y_pred):
    return 0.5 * np.sum(np.square(y_true - y_pred), axis=-1)

def derivative_mse(y_true, y_pred):
    return y_pred - y_true

def update_parameters(parameters, gradient, learning_rate):
    return parameters - learning_rate*gradient

# ReLU and its derivative
def relu(x):
    return x*(x > 0)

def derivative_relu(x):
    return 1*(x > 0)

def forward_pass(x, w1, b1, w2, b2):
    layer1 = relu(np.dot(x, w1) + b1)
    y_hat = np.dot(layer1, w2.transpose()) + b2
    return y_hat

def cal_gradient(x, y, y_hat, w1, b1, w2, b2):
    ly1 = np.dot(x, w1) + b1
    dLdw1 = derivative_mse(y, y_hat) * derivative_relu(ly1) * x * w2.reshape(1, -1)
    dLdb1 = derivative_mse(y, y_hat) * derivative_relu(ly1) * w2.reshape(1, -1)
    dLdw2 = derivative_mse(y, y_hat) * relu(ly1)
    dLdb2 = derivative_mse(y, y_hat)
    return dLdw1, dLdb1, dLdw2, dLdb2


class NeuralNetwork():
    def __init__(self, parameters, learning_rate, step):
        self.w1, self.b1, self.w2, self.b2 = parameters
        self.learning_rate = learning_rate
        self.step = step
        self.losses = []

    def train(self, x, y, epochs):
        for e in range(epochs):
            # forwardpass
            y_hat = forward_pass(x, self.w1, self.b1, self.w2, self.b2)
            loss = np.mean(mse(y, y_hat))
            self.losses.append(loss)

            # print training procedure
            if (e + 1) % self.step == 0:
                print('[EPOCH %d/%d] loss: %.4f' % (e + 1, epochs, loss))

            # calculate gradient
            dLdw1, dLdb1, dLdw2, dLdb2 = cal_gradient(x, y, y_hat, self.w1, self.b1, self.w2, self.b2)

            # calculate gradient over entire dataset
            dLdw1 = np.mean(dLdw1, axis=0, keepdims=True)
            dLdb1 = np.mean(dLdb1, axis=0, keepdims=True)
            dLdw2 = np.mean(dLdw2, axis=0, keepdims=True)
            dLdb2 = np.mean(dLdb2, axis=0)

            # update parameters
            self.w1 = update_parameters(self.w1, dLdw1, self.learning_rate)
            self.b1 = update_parameters(self.b1, dLdb1, self.learning_rate)
            self.w2 = update_parameters(self.w2, dLdw2, self.learning_rate)
            self.b2 = update_parameters(self.b2, dLdb2, self.learning_rate)

    def predict(self, x):
        return forward_pass(x, self.w1, self.b1, self.w2, self.b2)


def initialize_parameters(method, num_neurons):
    random_state = np.random.RandomState(42)

    # bias are initialized as zeros for all experiments
    b1 = np.zeros(shape=(1, num_neurons))
    b2 = 0

    if method in ['normal']:
        stddev = 0.1
        w1 = stddev * random_state.randn(1, num_neurons)
        w2 = stddev * random_state.randn(1, num_neurons)
    elif method in ['uniform']:
        limit = 0.1
        w1 = random_state.uniform(-limit, limit, size=(1, num_neurons))
        w2 = random_state.uniform(-limit, limit, size=(1, num_neurons))
    else:
        raise ValueError('Please enter a valid initialization method...')

    return w1, b1, w2, b2


x, y = create_toy_dataset(n_samples=100)
loss_results = []
prediction_results = []
for init_method in ['normal', 'uniform']:
    print('----- initialization method: %s -----' % init_method)
    # initialize network parameters
    N_NEURONS = 500
    parameters = initialize_parameters(init_method, N_NEURONS)

    # build and train a network
    nn = NeuralNetwork(parameters=parameters, learning_rate=1e-2, step=500)
    nn.train(x, y, epochs=5000)

    loss_results.append(nn.losses)
    prediction_results.append(nn.predict(x))

# plot
plt.figure(figsize=(5, 5))
plt.plot(x, prediction_results[0], label='normal')
plt.plot(x, prediction_results[1], label='uniform')
plt.plot(x, y, label='ground truth')
plt.xticks(ticks=np.linspace(-1, 1, 5), labels=np.linspace(-1, 1, 5))
plt.legend()
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(loss_results[0], label='normal')
plt.plot(loss_results[1], label='uniform')
plt.ylim([0, 0.1])
plt.legend()
plt.show()


