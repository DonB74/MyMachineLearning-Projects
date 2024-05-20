# J74
# Don Binoy; 22209158

# Problem Description

# Select a dataset with multiple features suitable for regression analysis.
# Choose a single feature as your predictor and split the dataset into a large test set (50%) and the remaining data for training.
# Starting with a very small training subset, train a linear regression model and calculate the Residual Sum of Squares (RSS) on the test data.
# Gradually increase the training subset size, retraining the model and recalculating the RSS at each step.
# Plot the RSS against the train data size to visualize the relationship.
# Finally, for a chosen train data size, create scatter plots showing the regression line and data points for both the training and test sets
# to observe how the model's generalization improves with more training data.

# Source of dataset(auto.csv): https://www.statlearning.com/s/ALL-CSV-FILES-2nd-Edition-corrected.zip

# Importing important modules
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# %matplotlib inline

# Loading the auto.csv data
df = pd.read_csv("Auto.csv")
print(df.head())

# Initialization of the plots
fig, axes = subplots(ncols=2, nrows=5, figsize=(33, 74))

# Plotting all possibles graphs inorder to identify which pair is suitable for regression

# mpg vs displacement
axes[0,0].set_xlabel('mpg')
axes[0,0].set_ylabel('displacement')
axes[0,0].scatter(df['mpg'], df['displacement'])

# mpg vs horsepower
axes[0,1].set_xlabel('mpg')
axes[0,1].set_ylabel('hp')
axes[0,1].scatter(df['mpg'], df['horsepower'])

# mpg vs weight
axes[1,0].set_xlabel('mpg')
axes[1,0].set_ylabel('weight')
axes[1,0].scatter(df['mpg'], df['weight'])

# mpg vs acceleration
axes[1,1].set_xlabel('mpg')
axes[1,1].set_ylabel('acceleration')
axes[1,1].scatter(df['mpg'], df['acceleration'])

# displacement vs horsepower
axes[2,0].set_xlabel('displacement')
axes[2,0].set_ylabel('hp')
axes[2,0].scatter(df['displacement'], df['horsepower'])

# displacement vs weight
axes[2,1].set_xlabel('displacement')
axes[2,1].set_ylabel('weight')
axes[2,1].scatter(df['displacement'], df['weight'])

# displacement vs acceleration
axes[3,0].set_xlabel('displacement')
axes[3,0].set_ylabel('acceleration')
axes[3,0].scatter(df['displacement'], df['acceleration'])

# horsepower vs weight
axes[3,1].set_xlabel('hp')
axes[3,1].set_ylabel('weight')
axes[3,1].scatter(df['horsepower'], df['weight'])

# horsepower vs acceleration
axes[4,0].set_xlabel('hp')
axes[4,0].set_ylabel('acceleration')
axes[4,0].scatter(df['horsepower'], df['acceleration'])

# weight vs acceleration
axes[4,1].set_xlabel('weight')
axes[4,1].set_ylabel('acceleration')
axes[4,1].scatter(df['weight'], df['acceleration'])

# Selecting displacement vs weight for linear regression
# random splitting of data into train and test datasets

# Assigning the features into variables
X = df['displacement']
y = df['weight']

# initializing the plot
fig, ax = subplots(figsize=(14,14))
ax.scatter(X, y)

# Splitting into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshaping the X values inorder to fit in linear regression by sklearn
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

lin_reg = LinearRegression().fit(X_train, y_train)

# Prediction of weights
y_pred = lin_reg.predict(X_test)
print(y_pred)

# Plotting the regression line
ax.plot(X_test, y_pred)

# Plotting RSS against train size

RSS = []
test_size = [i/100 for i in range(10,100,10)]
fig, ax = subplots(figsize=(24,18))


for size in test_size:

  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=size)
  train_X = train_X.values.reshape(-1, 1)
  RSS.append(mean_squared_error(test_X, test_y)*len(test_X))

RSS = np.asanyarray(RSS)
test_size = np.asanyarray(test_size)

ax.set_xlabel("Residual Sum of Squares")
ax.set_ylabel("Train Size")
ax.scatter(RSS, 1 - test_size)
ax.plot(RSS, 1 - test_size, 'orange')
