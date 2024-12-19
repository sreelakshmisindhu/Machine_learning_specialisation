import numpy as np
import matplotlib.pyplot as plt
from common_functions import sigmoid_function

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w = np.zeros((1))
b = 0

x_train_sigmoid = sigmoid_function(x_train*w +b)
plt.scatter(x_train, x_train_sigmoid)
plt.show()
