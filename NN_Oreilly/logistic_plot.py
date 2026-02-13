from sympy import *
import matplotlib.pyplot as plt
import numpy as np

#1. formula
x=symbols('x')
logistics = 1/ (1+ exp(-x))
#2. convert
f=lambdify(x, logistics, "numpy")
#3. data 
x_vals = np.linspace(-10, 10, 100)
y_vals = f(x_vals)
#4. plot 
# 4. Plot and Save (Matplotlib)
plt.plot(x_vals, y_vals)
plt.title("Logistic Activation Function")
plt.xlabel("Input (x)")
plt.ylabel("Output (0, 1)")
plt.grid(True)

# Save the file
plt.savefig("Logistic_plot.png")
print("Success! Image saved as Logistic_plot.png")