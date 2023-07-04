import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming the dataset is in a CSV file named 'power_plant.csv'
data = pd.read_excel('PowerPlant.xlsx')

# Assuming the target variable is named 'output' and the features are in columns 'feature1', 'feature2', etc.
X = data[['AT', 'V', 'AP', 'RH']]  # Features
y = data['PE']  # Target variable

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print('Mean Squared Error (MSE):', mse)
print('R-squared score:', r2)
