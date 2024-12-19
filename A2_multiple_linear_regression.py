import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from common_functions import *

"""
linear regression with four input features and scaling.
"""

#load input data 
x_train, y_train = load_data()
x_features = ['size(sqft)','bedrooms','floors','age']
x_mean = np.mean(x_train,axis=0)
x_sigma = np.std(x_train,axis=0)
plot_all(x_train, y_train, x_features)

#apply z score scaling
x_train_norm = scaling(x_train, 0, "sigma")
x_train_norm = scaling(x_train, 1, "sigma")
#print(f"x_train2_norm{x_train2_norm}, axis ")
x_train_norm = scaling(x_train, 2, "sigma")
x_train_norm = scaling(x_train, 3, "sigma")
norm_compare_plot(x_train, x_train_norm, x_features)

#input data
#x_train = np.array([[2104, 5, 1, 45],[1416, 3, 2, 40],[852, 2, 1, 35]])
#y_train = np.array([460, 232, 178])
#print(f"x_train = {x_train}")
#print(f"y_train = {y_train}")
#print(f"number of entries: {len(x_train)}")
#i = 1
#x_i = x_train[i]
#y_i = y_train[i]
#print(f"(x({i}), y({i})) = ({x_i}, {y_i})")


#fit parameters
#w = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
#b = 785.1811367994083
w = np.array([0,0,0,0])
b = 0
print(f"w: {w}")
print(f"b: {b}")
cost_value = cost_function(x_train_norm, y_train, w, b) #calculate cost
print(cost_value)
costs, new_w, new_b, j = gradient(x_train_norm, y_train, w, b, 1e-1,1000) #apply gradient descent
#print(gradient_info)
test_prediction(x_train, y_train, x_train_norm, x_features, new_w[-1], new_b[-1])



# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot gradient_w
axs[0, 0].plot(new_w[:, 0], costs[:], label='gradient_w', color='blue')
axs[0, 0].set_ylabel('Cost')
axs[0, 0].set_xlabel('w')
axs[0, 0].set_title('Cost vs w')


# Plot gradient_b
axs[0, 1].plot(new_b, costs, label='gradient_b', color='green')
axs[0, 1].set_ylabel('Cost')
axs[0, 1].set_xlabel('b')
axs[0, 1].set_title('Cost vs b')


# Plot cost_iter
axs[1, 0].plot(j, costs, label='cost_iter', color='red')
axs[1, 0].set_ylabel('Cost')
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_title('Cost vs Iteration')


# Plot data points and the linear function
#new_f_wb = linear_function(x_train, new_w[-1], new_b[-1])
print(f"plotting final {x_train}, {new_w[-1]}, {new_b}")
axs[1, 1].scatter(x_train_norm[:, 0], y_train, marker='x', c='r', label='Data')
#axs[1, 1].plot(x_train, new_f_wb, c='b', label='Prediction')
axs[1, 1].set_title('Housing Prices')
axs[1, 1].set_ylabel('Price (in 1000s of dollars)')
axs[1, 1].set_xlabel('Size (1000 sqft)')
axs[1, 1].legend()

# Adjust layout and save
#fig.tight_layout()
fig.savefig('combined_plots.png', dpi=300)

#predicting new values
x_required = np.array([1200, 3, 1, 40])
x_required_norm = (x_required - x_mean)/x_sigma
print(x_required_norm) 
price = linear_function(x_required_norm, new_w[-1], new_b[-1])
print(f"The values of w and b from gradient descent are {new_w[-1]} and {new_b[-1]}.")
#price = linear_function(x_required,gradient_info[-1,0], gradient_info[-1,1])
#print(f"The values of w and b from gradient descent are {gradient_info[-1,0]} and {gradient_info[-1, 1]}")
#
#
print(f"Price for 40 year old, 1200 sqft house with 3 bedrooms, 1 floors is {price:.0f} thousand dollars")


