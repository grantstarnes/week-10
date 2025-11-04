# Importing necessary libraries, pickle for saving the model, pandas for data handling, LinearRegression from sklearn
# Added numpy and DecisionTreeRegressor for Exercise 2
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# --------------------- Exercise 1 ---------------------

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

# --------------------- Exercise 2 ---------------------

# Map roast categories to numerical values
def roast_category(roast):
    '''
    This function roast_category takes in a roast string and returns a 
    numerical value based on the defined mapping (0-4)
    '''
    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }

    # Returns the mapped value or NaN if roast not found
    return mapping.get(roast, np.nan)

# Apply the roast_category function to create a new column 'roast_cat'  in the DataFrame
df_coffee["roast_cat"] = df_coffee["roast"].apply(roast_category)

# Set up the feature matrix X (100g_USD and roast_cat) and target vector y (rating)
X = df_coffee[["100g_USD", "roast_cat"]]
y = df_coffee["rating"]

# Create and train the Decision Tree Regressor model
dtr = DecisionTreeRegressor(random_state=42)

dtr.fit(X, y)

# Save the trained model to a file named "model_2.pickle"
with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)

# Print confirmation message that the model has been trained and saved
print("Model trained and saved as model_2.pickle")