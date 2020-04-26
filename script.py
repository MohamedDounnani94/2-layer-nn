import numpy as np
import matplotlib.pyplot as plt

costs = []
outputs = []

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

# 2 layer neural network
class NeuralNetwork:
  def __init__(self, x, y):
    # input
    self.input = x
    # desired output
    self.y = y
    # initial random weights [[r r r r], [r r r r], [r r r r]]
    self.weights_1 = np.random.rand(self.input.shape[1], 4)
    # [[r], [r], [r], [r]]
    self.weights_2 = np.random.rand(4, 1)
    self.output = np.zeros(self.y.shape)

  def feedforward(self):
    # value of each node of layer 1
    self.layer1 = sigmoid(np.dot(self.input, self.weights_1))
    # value of output
    self.output = sigmoid(np.dot(self.layer1, self.weights_2))

  def backprop(self):
    loss = np.power((self.output - self.y), 2)
    costs.append(np.mean(loss))
    d_cost = (2*(self.y - self.output) * sigmoid_derivative(self.output))
    d_weights_2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
    d_weights_1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights_2.T) * sigmoid_derivative(self.layer1)))
    self.weights_1 += d_weights_1
    self.weights_2 += d_weights_2


X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0],[1],[1],[0]])

n = 15000
nn = NeuralNetwork(X,y)
for i in range(n):
    nn.feedforward()
    nn.backprop()


print('Output\n')
print(nn.output)

plt.plot(np.arange(n), costs)
plt.xlabel('Iteraction')
plt.ylabel('Loss')

plt.show()