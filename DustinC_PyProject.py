# 01 May 2023
# ISM 4300 Project
# Dustin Cooksey
# This project takes a data set of housing prices in California
# and attempts to predict the value of the house using two different
# models based on the sets of parameters in the data set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# This is where the file is read into a data frame
df = pd.read_csv("california_housing.csv")

# Prints the data frame and shows information about it
print(df.head(10))

print(df.info())

# This is where the response are feature variables are split
# into two different data frames
df_response = df.copy(deep=True)
df_features = df.copy(deep=True)

# The X and Y variables are declared and assigned their approprate columns
x = df_features.drop(['median_house_value'], axis=1)
y = df_response['median_house_value']

# The data is split into the appropriate training and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# This is where a histogram is plotted of the data for visualization
df_response.hist(figsize=(30,20))

# This is where a heatmap of the variables is plotted 
# to the the corresponding correlations
plt.figure(figsize=(15,8))
sns.heatmap(df_features.corr(), annot=True, cmap="YlGnBu")

# This is where the scatter plot is plotted of the locations of the houses
plt.figure(figsize=(15,8))
sns.scatterplot(x="latitude", y="longitude", data=df_features, hue='median_house_value', palette="coolwarm")

# Here is where the regression model is fitted with the training data
linear_regress = LinearRegression()
linear_regress.fit(X_train, y_train)

# This is where the analytics are printed for the Regression Model
print("R^2 =", linear_regress.score(X_train, y_train))

print("The intercept (b_0) is {}".format(linear_regress.intercept_))
print("The coef (b_1) is {}".format(linear_regress.coef_[0]))
print("The coef (b_2) is {}".format(linear_regress.coef_[1]))

# This is where the prediction is tested for the regression model
price_pred = linear_regress.predict(X_test)
print(price_pred)

# Here is where the results for the Regression model are shown
print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, price_pred))
print("Mean Squared Error: ", metrics.mean_absolute_error(y_test, price_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, price_pred)))
 
# This is where the random forest is assigned the training data and fitted    
forest = RandomForestRegressor()
forest.fit(X_train, y_train)

# This is where the random forest score is shown
print("Forest score is: " ,forest.score(X_test, y_test))

