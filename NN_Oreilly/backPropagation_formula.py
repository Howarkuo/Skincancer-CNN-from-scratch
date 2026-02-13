# input (x) -> z1 (raw signal) -> A1 (filtered signal) -> z2 (final raw score) -> A2 (final probability)
# Cost of Prediction C
# C = (A2 -Y) **2
from sympy import *
import numpy as np

W1, W2, B1,B2, A1,A2, Z1,Z2, X,Y = \
    symbols('W1, W2, B1,B2, A1,A2, Z1,Z2, X,Y')

# Cost to A2 differentiation
A2, Y = symbols ('A2 Y')
C= (A2 -Y) **2
dC_dA2 = diff(C,A2)
print (dC_dA2)
# 2*A2 - 2*Y

# A2 to Z2 differentiation
Z2= symbols ('Z2')
logistic = lambda x: 1 / (1+exp(-x))
A2= logistic(Z2)
dA2_dZ2 = diff(A2, Z2)
print(dA2_dZ2)
# exp(-Z2)/(1 + exp(-Z2))**2

# Z2 to W2  differentiation

A1, W2, B2 = symbols("A1, W2, B2")

Z2= A1*W2+B2
dZ2_dW2 = diff(Z2, W2)
# diff function: the function, variable with respect to
print(dZ2_dW2)
#A1


# A1 to Z1 differentiation
# relu= lambda x: np.maximum(x,0)
A1_eq =Max(0,Z1)
# _A1 =  relu(Z1)
# d_relu= lambda x: x>0
dA1_dZ1 = diff(A1_eq,Z1)
print('dA1_dZ1',dA1_dZ1)

# Z1 to W1  differentiation

Z1_eq= X*W1 + B1
dZ1_dW1= diff(Z1_eq, W1)
print ('dZ1_dW1',dZ1_dW1)


#Z1 to B1 differentiation
dZ1_dB1= diff(Z1_eq, B1)
print ('dZ1_dB1',dZ1_dB1)
