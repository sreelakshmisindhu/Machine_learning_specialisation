import numpy as np
from common_functions import *

"""
Implements polynomial regression

"""
#Create a sample dataset which is not linear (cosine)
x_train = np.arange(0.0, 18.0, 1)
y_train = np.cos(x_train/2)
x_train_mod = np.c_[x_train, x_train**2, x_train**3, x_train**4, x_train**5, x_train**6, x_train**7, x_train**8, x_train**9, x_train**10, x_train**11, x_train**12, x_train**13, x_train**14, x_train**15]

#apply scaling for the new features at higher order
for i in range(x_train_mod.shape[1]):
    x_train_norm = scaling(x_train_mod, i, "sigma") 

print(x_train_norm)

w = np.zeros(15)
b=0
costs, new_w, new_b, iter_n = gradient(x_train_norm, y_train, w, b, 0.1, 50000)


#plt.scatter(iter_n, costs)
plt.scatter(x_train, y_train)
plt.plot(x_train, x_train_norm@new_w[-1]+ new_b[-1])
plt.title("test prediction")
plt.show()
