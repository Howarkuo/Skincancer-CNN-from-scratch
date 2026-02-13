
# logits -> probability (ReLU) -> loss function
# Z= WX + B
# input (x) -> z1 (raw signal) -> A1 (filtered signal) -> z2 (final raw score) -> A2 (final probability)
# forward passin
# 1 is light
# 0 is dark
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

all_data= pd.read_csv("https://tinyurl.com/y2qmhfsr")

# RED,GREEN,BLUE,LIGHT_OR_DARK_FONT_IND
# 0,0,0,0
# 0,0,128,0
# 0,0,139,0
# 0,0,205,0
# 0,0,238,0
#......
all_inputs= (all_data.iloc[:,0:3].values/255.0)
all_outputs = all_data.iloc[:, -1].values

X_train, X_test, Y_train, Y_test=train_test_split(all_inputs,all_outputs,test_size=1/3)

n=X_train.shape[0]

# randomized initiation
w_hidden=np.random.rand(3,3)
w_output=np.random.rand(1,3)


b_hidden=np.random.rand(3,1)
b_output=np.random.rand(1,1)

relu= lambda x: np.maximum(x,0)
logistic= lambda x: 1/ (1+np.exp(-x))

def forward_prop(x):
    Z1= w_hidden @ x + b_hidden
    A1= relu(Z1)
    Z2 = w_output @A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

testpredictions= forward_prop(X_test.transpose())[3]
testcomparisons= np.equal((testpredictions >= .5).flatten().astype(int),Y_test)

accuracy = sum(testcomparisons.astype(int)/X_test.shape[0])

print("accuracy",accuracy)


# accuracy 0.639198218262805


# --- NEW VISUALIZATION SECTION ---
print("Generating decision boundary plot...")

# 1. Create a grid of points to "scan" the network's decision making
# We will vary Red and Green from 0 to 1, and fix Blue at 0.5
x_span = np.linspace(0, 1, 200) # Red
y_span = np.linspace(0, 1, 200) # Green
xx, yy = np.meshgrid(x_span, y_span)

# Flatten them to creating a long list of inputs
red_flat = xx.ravel()
green_flat = yy.ravel()
blue_fixed = np.full(red_flat.shape, 0.5) # Fix Blue at 0.5 for the slice

# Stack into the shape (3, N) that forward_prop expects
grid_input = np.vstack((red_flat, green_flat, blue_fixed))

# 2. Ask the network for predictions on this dummy grid
# We grab index [3] which is A2 (Final Probability)
_, _, _, grid_predictions = forward_prop(grid_input)

# Reshape predictions back to the 2D grid shape (200x200)
Z_grid = grid_predictions.reshape(xx.shape)

# 3. Plotting
plt.figure(figsize=(10, 8))

# A. Draw the Background (The Network's "Mind")
# Red area = Network predicts 0 (Dark Font)
# Blue area = Network predicts 1 (Light Font)
contour = plt.contourf(xx, yy, Z_grid, levels=50, cmap="RdBu", vmin=0, vmax=1, alpha=0.6)
plt.colorbar(contour, label="Network Prediction (Probability)")

# B. Draw the Actual Data Points (The Truth)
# We overlay the Test Data to see where the network is right or wrong
# We only plot points where Blue is roughly near 0.5 to keep the slice accurate
mask = (X_test[:, 2] > 0.4) & (X_test[:, 2] < 0.6)
subset_X = X_test[mask]
subset_Y = Y_test[mask]

# Scatter plot: 
# Dots are colored by the REAL answer (Y_test). 
# If a Black Dot is in a Blue Region, the network is wrong!
plt.scatter(subset_X[:, 0], subset_X[:, 1], c=subset_Y, cmap="RdBu_r", edgecolors='k', s=50, label="Actual Data")

plt.title(f"Network Decision Boundary (Random Weights)\nAccuracy: {accuracy:.2f}")
plt.xlabel("Red Intensity")
plt.ylabel("Green Intensity")
plt.legend(["Actual Data (Dark=0, Light=1)"])
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig("decision_boundary_random.png")
print("Plot saved as 'decision_boundary_random.png'")
plt.show()