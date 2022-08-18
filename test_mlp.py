import numpy as np
import pandas as pd
from numpy import load
from acc_calc import accuracy
import pickle

def test_mlp(data_file):
    # Load the test set
    test_features = pd.read_csv(data_file)
    test_features = np.array(test_features.T)
    # activation function
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    # activation function for output layer
    def softmax(A):
        expA = np.exp(A)
        return expA / expA.sum(axis=0, keepdims=True)
    # Function for forward propagation ( containing dot product and activation function)
    def forw_prop(X, params):

        Z1= np.dot(params["Weight1"],X) + params["bias1"] 
        Activ_func1= sigmoid(Z1)
        Z2= np.dot(params["Weight2"],Activ_func1) + params["bias2"]
        Activ_func2= softmax(Z2)
        deravatives = {"Z1": Z1,
            "Activ_func1": Activ_func1,
            "Z2": Z2,
            "Activ_func2": Activ_func2 }
    
        return deravatives
    
    # Converting to one-hot encodings for final predicted output
    def encode(y):   
        li = list()
        for i in y:
            temp = [0, 0, 0, 0]
            temp[i] = 1
            li.append(temp)
        
        return np.array(li)
    
    # loading weights from pickle file and applying forward propagation for making predictions
    f = open('weights', 'rb')
    params = pickle.load(f)
    f.close()
    deravatives = forw_prop(test_features, params)
    predictions = np.argmax(deravatives["Activ_func2"], axis=0)
    y_pred = encode(predictions)

    return y_pred
'''
How we will test your code:

from test_mlp import test_mlp
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''