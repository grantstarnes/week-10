# Importing necessary libraries, pickle for saving the model, pandas for data handling, and LinearRegression from sklearn
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the coffee analysis data from the url into a pandas DataFrame
df_coffee = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

# Set up the feature matrix X (100g_USD) and target vector y (rating)
X = df_coffee[["100g_USD"]]
y = df_coffee["rating"]

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file named "model_1.pickle"
with open("model_1.pickle", "wb") as f:
    pickle.dump(model, f)

# Print confirmation message that the model has been trained and saved
print("Model trained and saved as model_1.pickle")