from sklearn.linear_model import LinearRegression
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load the data
X_frac = np.load("TrainingData/X_frac.npy")
y = np.load("TrainingData/y.npy")

frac = np.load("TrainingData/frac_col.npy", allow_pickle=True) # allow_pickel = True when you input string data

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_frac, y, test_size=0.2, random_state=42)

# Load the model from the saved file
with open("Models/frac_xgboost.pkl", "rb") as model_file:
 model = pickle.load(model_file)
 
# Now you can use the loaded model for predictions
y_pred = model.predict(X_test)

# Calculate the mean squared error (MSE) as a measure of model performance
mse = mean_squared_error(y_test, y_pred)
mse_text = f"Mean Squared Error: {mse:.2f}"

# Calculate the R2 score
r2 = r2_score(y_test, y_pred)
r2_text = f"R2 Score: {r2:.2f}"

# Calculate the slope and intercept of the line that best fits the data
lr = LinearRegression()
lr.fit(y_test.reshape(-1, 1), y_pred)
slope, intercept = lr.coef_[0], lr.intercept_

# Calculate the absolute differences
diff = np.abs(y_test - y_pred)

# Normalize the absolute differences
norm = plt.Normalize(vmin=0, vmax=np.max(diff))

# Create a colormap and use it to map the absolute differences to colors
cmap = plt.cm.get_cmap('hot')
colors = cmap(norm(diff))

# Create a graph for prediction vs. actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c=colors, alpha=0.5)
plt.plot(y_test, slope * y_test + intercept, linestyle='dashed')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Prediction vs. Actual Values")

# Add the MSE and R2 values as text in the graph
plt.text(0.05, 0.7, mse_text, transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.05, 0.8, r2_text, transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.05, 0.9, "Element Fraction", transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel('|Actual - Predicted|', rotation=270, fontsize=15, labelpad=15)

plt.savefig("Images/frac_xgboost.png") # Save the plot to a file
print("Success Save")
plt.show()
