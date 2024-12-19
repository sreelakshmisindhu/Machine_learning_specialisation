import numpy as np
import matplotlib.pyplot as plt
from common_functions import linear_function, cost_function

"""
This code implements linear regression with one input parameter.
"""

def gradient(x, y, w, b, a, num_iter):
    """
    This function returns the modified values of w and b and the curresponding costs as a np array of the form [[w,b,cost]]
    it takes x,y intial values of w, b, stepsize a and number of iterations as input.
    """
    costs =[]
    tmp_w = w
    tmp_b = b
    for j in range(num_iter):
        dw = 0
        db = 0
        for i in range(len(x)):    
            dw =dw + (linear_function(x, tmp_w, tmp_b)[i]-y[i])*x[i]
            db =db + linear_function(x, tmp_w, tmp_b)[i]-y[i] 
            #print(dw, db, linear_function(x, tmp_w, tmp_b)[i])
        tmp_b =tmp_b - a*(db/len(x))
        tmp_w =tmp_w - a*(dw/len(x))
        cost = cost_function(x, y, tmp_w, tmp_b)
        costs.append([tmp_w, tmp_b, cost, j])
        print(tmp_w, tmp_b, cost, j)
    costs =np.array(costs) 
    return costs
    

#input data
x_train = np.array([1.0, 2.0])
y_train = np.array([300., 500.])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
print(f"number of entries: {len(x_train)}")
i = 1
x_i = x_train[i]
y_i = y_train[i]
print(f"(x({i}), y({i})) = ({x_i}, {y_i})")


#fit parameters
w = 0
b = 0
print(f"w: {w}")
print(f"b: {b}")
cost_value = cost_function(x_train, y_train, w, b)
print(cost_value)
gradient_info = gradient(x_train, y_train, w, b, 0.01,100000)
print(gradient_info)



# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot gradient_w
axs[0, 0].plot(gradient_info[:, 0], gradient_info[:, 2], label='gradient_w', color='blue')
axs[0, 0].set_ylabel('Cost')
axs[0, 0].set_xlabel('w')
axs[0, 0].set_title('Cost vs w')


# Plot gradient_b
axs[0, 1].plot(gradient_info[:, 1], gradient_info[:, 2], label='gradient_b', color='green')
axs[0, 1].set_ylabel('Cost')
axs[0, 1].set_xlabel('b')
axs[0, 1].set_title('Cost vs b')


# Plot cost_iter
axs[1, 0].plot(gradient_info[:, 3], gradient_info[:, 2], label='cost_iter', color='red')
axs[1, 0].set_ylabel('Cost')
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_title('Cost vs Iteration')


# Plot data points and the linear function
new_f_wb = linear_function(x_train, gradient_info[-1, 0], gradient_info[-1, 1])
axs[1, 1].scatter(x_train, y_train, marker='x', c='r', label='Data')
axs[1, 1].plot(x_train, new_f_wb, c='b', label='Prediction')
axs[1, 1].set_title('Housing Prices')
axs[1, 1].set_ylabel('Price (in 1000s of dollars)')
axs[1, 1].set_xlabel('Size (1000 sqft)')
axs[1, 1].legend()

# Adjust layout and save
#fig.tight_layout()
fig.savefig('combined_plots.png', dpi=300)
plt.show()


#predicting new values
x_required = np.array([1.2])
price = linear_function(x_required,gradient_info[-1,0], gradient_info[-1,1])
print(f"The values of w and b from gradient descent are {gradient_info[-1,0]} and {gradient_info[-1, 1]}")


print(f"Price for 1200 sqft is {price[0]:.0f} thousand dollars")


