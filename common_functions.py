import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#linear regression example with two data points

def linear_function(x, w, b):
    """
    This function returns wx +b for a given array of x and fixed values of w and b 
    """
    f_wb = np.zeros(len(x))
    f_wb = np.dot(x,w) + b
    return f_wb

def cost_function(x, y, w, b):
    """
    This function computes the squared error cost for given array of y from f(x)
    """
    cost = 0
    f_wb = linear_function(x, w, b)
    for i in range(len(x)):
        cost = cost+(f_wb[i]-y[i])**2
    cost = cost/(2*len(x)) 
    return cost

def gradient(x, y, w, b, a, num_iter):
    """
    This function returns the modified values of w and b and the curresponding costs as a np array of the form [[w,b,cost]]
    it takes x,y intial values of w, b, stepsize a and number of iterations as input.
    """
    costs = np.zeros((num_iter,))
    new_b = np.zeros((num_iter,))
    new_w = np.zeros((num_iter, w.shape[0]))
    iter_n = np.zeros((num_iter,)) 
    tmp_w = w
    tmp_b = b
    for j in range(num_iter):
        dw = np.zeros(w.shape[0])
        db = 0
        for i in range(len(x)):    
            dw =dw + (linear_function(x, tmp_w, tmp_b)[i]-y[i])*x[i]
            db =db + linear_function(x, tmp_w, tmp_b)[i]-y[i] 
            #print(dw, db, linear_function(x, tmp_w, tmp_b)[i])
        tmp_b =tmp_b - a*(db/len(x))
        tmp_w =tmp_w - a*(dw/len(x))
        cost = cost_function(x, y, tmp_w, tmp_b)
        costs[j] = cost
        new_w[j] = tmp_w
        new_b[j] = tmp_b
        iter_n[j] = j

        print(f"w, b, cost^, j{tmp_w}, {tmp_b}, {cost}, {j}")
    #costs =np.array(costs) 
    return costs, new_w, new_b, iter_n 

def load_data():
    """
    This functions loads data of the form x1,x2,x3,x4,y. can be modified for different number of inputs
    """

    data = np.loadtxt("houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y    

def scaling(x, column, type_of):
    """
    This function applies min_max, mean and Z score scaling. The scaling can be applied for only one row at a time for more flexibiity
    """
 
    highest = np.max(x,axis=0)
    lowest = np.min(x,axis=0)
    mean = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    if type_of == "min_max":
        x[:, column] = x[:, column]/highest[column]
    if type_of == "mean":
        x[:, column] = (x[:, column]-mean)/(highest-lowest)
    if type_of == "sigma":
        x[:, column] = (x[:, column]-mean[column])/sigma[column]
    print(highest, " highest ", lowest, " lowest  ", mean, " mean  ", sigma, " sigma  ")
    print("x: ", x)
    return x
    
def plot_all(x_train, y_train, x_features):
    """
    This function scatter plots all the 4 input features
    """

    fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x_train[:,i],y_train)
        ax[i].set_xlabel(x_features[i])
    ax[0].set_ylabel("Price (1000's)")
    fig.savefig("data_visualisation.png")    

def norm_plot(ax, data):
    """
    This function makes a normal distribution 
    """
    scale = (np.max(data) - np.min(data))*0.2
    x = np.linspace(np.min(data)-scale,np.max(data)+scale,50)
    _,bins, _ = ax.hist(data, x, color="xkcd:azure")
    #ax.set_ylabel("Count")
    
    mu = np.mean(data); 
    std = np.std(data); 
    dist = norm.pdf(bins, loc=mu, scale = std)
    
    axr = ax.twinx()
    axr.plot(bins,dist, color = "orangered", lw=2)
    axr.set_ylim(bottom=0)
    axr.axis('off')


def norm_compare_plot(x_train, x_train_norm, x_features):
    """
    compare the input features before and after normalisation
    """
    fig,ax=plt.subplots(1, 4, figsize=(12, 3))
    for i in range(len(ax)):
        norm_plot(ax[i],x_train[:,i],)
        ax[i].set_xlabel(x_features[i])
    ax[0].set_ylabel("count");
    fig.suptitle("distribution of features before normalization")
    plt.show()
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    for i in range(len(ax)):
        norm_plot(ax[i],x_train_norm[:,i],)
        ax[i].set_xlabel(x_features[i])
    ax[0].set_ylabel("count"); 
    fig.suptitle("distribution of features after normalization")    
    fig.savefig("compare_normalisation.png")

def test_prediction(x_train, y_train, x_train_norm,  x_features, w, b):
    """
    Overlays the original values and predictions using w, and b from the linear regression 
    """

    m = x_train_norm.shape[0]
    yp = np.zeros(m)
    for i in range(m):
        print(x_train_norm[i], "  ", w)
        yp[i] = np.dot(w, x_train_norm[i]) + b
    
        # plot predictions and targets versus original features    
    fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(x_features[i])
        ax[i].scatter(x_train[:,i],yp,color="red", label = 'predict')
    ax[0].set_ylabel("Price"); ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    fig.savefig("testing_predictions.png")
  

