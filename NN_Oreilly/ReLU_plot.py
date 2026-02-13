# Rectified Linear Unit
# use matplotlib and lamdify to transform and save pic
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
#1.Return which is higher , 0 or value x
x= symbols('x')
relu=Max(0,x)
#2. convert sympy to python functiob
f= lambdify(x, relu, 'numpy')

#3. generate data
x_vals = np.linspace(-10, 10, 100)
y_vals = f(x_vals)


# 4. Plot and Save (Matplotlib)
plt.plot(x_vals, y_vals)
plt.title("ReLU Activation Function")
plt.xlabel("Input (x)")
plt.ylabel("Output (Max(0, x))")
plt.grid(True)

# Save the file
plt.savefig("relu_plot.png")
print("Success! Image saved as relu_plot.png")