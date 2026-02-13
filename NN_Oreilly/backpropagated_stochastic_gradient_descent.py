# input (x) -> z1 (raw signal) -> A1 (filtered signal) -> z2 (final raw score) -> A2 (final probability)
# Z= WX + B
# C = (A2 -Y)**2
# aim : dC /dW2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

all_data= pd.read_csv("https://tinyurl.com/y2qmhfsr")

from sympy import *
import numpy as np




L= 0.05

all_inputs= (all_data.iloc[:,0:3].values/255.0)
all_outputs = all_data.iloc[:, -1].values
X_train, X_test, Y_train, Y_test=train_test_split(all_inputs,all_outputs,test_size=1/3)

n=X_train.shape[0]

# randomized initiation
w_hidden=np.random.rand(3,3)
w_output=np.random.rand(1,3)


b_hidden=np.random.rand(3,1)
b_output=np.random.rand(1,1)
# lamba : anonymous function
relu= lambda x: np.maximum(x,0)
logistic= lambda x: 1/ (1+np.exp(-x))

def forward_prop(x):
    Z1= w_hidden @ x + b_hidden
    A1= relu(Z1)
    Z2 = w_output @A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

d_relu = lambda x: x >0
d_logistic = lambda x: np.exp(-x) / (1+np.exp(-x)) **2


def back_prop(Z1, A1, Z2, A2, X,Y,w_output):
    dC_dA2 = 2 * (A2 -Y)
    dA2_dZ2 = d_logistic(Z2)
    delta2= dC_dA2 * dA2_dZ2

    # layer 2 gradient
    dC_dW2 = delta2 @ A1.T
    dC_dB2 = delta2 
    # layer 1 error
    dZ2_dA1 = w_output
    dA1_dZ1 = d_relu(Z1)
    delta1 = (dZ2_dA1.T @ delta2) * dA1_dZ1

    # layer 1 gradients
    dC_dW1 = delta1 @ X.T
    dC_dB1= delta1



    # dZ2_dA1 = w_output
    # dZ2_dW2 = A1
    # dA1_dZ1 = d_relu(Z1)
    # dZ1_dW1 = X
    # dZ2_dB1 = 1

    # dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T
    # dC_dB2 = dC_dA2 @ dA2_dZ2 @ dZ2_dB2
    # dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1
    # dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T
    # dC_dB1 = dC_dA1 @ dA1_dZ1 @ dZ1_dB1
    return dC_dW1, dC_dB1, dC_dW2, dC_dB2
for i in range (100_000):
    idx = np.random.choice(n ,1 , replace= False)
    X_sample = X_train[idx].transpose()
    Y_sample = Y_train[idx]

    Z1, A1, Z2, A2 = forward_prop(X_sample)

    dW1, dB1, dW2, dB2 = back_prop(Z1, A1, Z2, A2, X_sample, Y_sample, w_output)
    w_hidden -= L *dW1
    b_hidden -= L *dB1
    w_output -= L *dW2
    b_output -= L *dB2
    if i % 10000 == 0:
        print(f"Step {i}, Accuracy check soon...")

testpredictions= forward_prop(X_test.transpose())[3]
testcomparisons= np.equal((testpredictions >= .5).flatten().astype(int),Y_test)
accuracy = sum(testcomparisons.astype(int)/X_test.shape[0])

print("accuracy",accuracy)
# accuracy 0.9910913140311729



def predict_prob(r,g,b):
    X= np.array([[r,g,b]]).transpose()/255
    Z1,A1,Z2,A2= forward_prop(X)
    return A2

def predict_font_shader(r,g,b):
    output_values = predict_prob(r,g,b)
    if output_values > .5:
        return "DARK"
    else:
        return "LIGHT"
    
while True:
    col_input = input ("Predicting light or dark font, Input value R,G,B")
    (r,g,b) = col_input.split(",")
    print(predict_font_shader(int(r),int(g),int(b)))

# Predicting light or dark font, Input value R,G,B 100, 20, 50
# LIGHT