
''' 
Author: Colby McClure
Date: 2/25/2024
Description: This program is a simple example of how to use the sklearn library to perform linear regression on the iris dataset.
Program: mainFunc.py 
Assignment: Homework 2 
'''

# Needed imports 
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the labels 
def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virginica'

# Function to create and show the heat map related to the iris data set 
def heatMap():

    # Create the correlation matrix
    numeric_cols_df = df.select_dtypes(include=[np.number])
    cor_eff = numeric_cols_df.corr(method='pearson')
    plt.figure(figsize=(6, 6))
    sns.heatmap(cor_eff, linecolor = 'white', linewidths = 1, annot = True)

    # Plot the lower half of the coorelation matrix
    fig, ax = plt.subplots(figsize=(6, 6))

    # Compute the correlation matrix
    mask = np.zeros_like(cor_eff) 

    mask[np.triu_indices_from(mask)] = 1
    sns.heatmap(cor_eff, linecolor = 'white', linewidths = 1, mask = mask, ax = ax, annot = True)

    plt.show() 

def pairPlot():
    # Visualize the data using pairplot // Uncomment to see the pairplot
    g = sns.pairplot(df, hue = 'species') 

    # Display the graph 
    plt.show() 


def linearReg(iris_df):
   
   # Drop the species column from the iris_df
   df.drop('species', axis = 1, inplace = True)
   target_df = pd.DataFrame(data = iris.target, columns = ['species'])

   # Concatenate the DataFrames
   iris_df = pd.concat([iris_df, target_df], axis = 1)

   # Split the data into training and testing sets
   X = iris_df.drop(labels = 'petal length (cm)', axis = 1)
   y = iris_df['petal length (cm)']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 121)
  
   # Load the linear regression model
   lr = LinearRegression()

   # Fit the model to the training data
   lr.fit(X_train, y_train)

   # Make predictions using the testing set
   lr.predict(X_test)
   y_pred = lr.predict(X_test)

    # Print the results
   print('LR beta/slope: ', lr.coef_)   

   print('LR alpha/slope intercept Coefficient: ', lr.intercept_)

   print('Coefficient of determination: ', r2_score(y_test, y_pred))

   print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)))

   print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))

   iris_df.loc[16] 

   # Create a dictionary of the data 
   d = {'sepal length (cm)': [5.4], 'sepal width (cm)': [3.9], 'petal length (cm)': [1.3], 'petal width (cm)': [0.4], 'species': [0]}

   # Create a DataFrame from the dictionary
   pred_df = pd.DataFrame(data = d)
   print(pred_df) 

   # Make a prediction using the linear regression model
   pred = lr.predict(X_test)
    
   # Output the predicted versus the actual petal length
   print('Predicted Petal Length (cm): ', pred[0])
   print('Actual Petal Length (cm): ', y_test.iloc[0]) 
   

# Fetch dataset 
iris = load_iris()

# Create the pandas DataFrames
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
target_df = pd.DataFrame(data = iris.target, columns = ['species'])


# Apply the converter
target_df['species'] = target_df['species'].apply(converter)

# Concatenate the DataFrames
df = pd.concat([iris_df, target_df], axis = 1)

# Mode = 1 for heat map, 2 for pair plot, 3 for linear regression
# Change this to see the different visualizations
mode = 3

if(mode == 1):
    heatMap()

elif(mode == 2):
    pairPlot()

elif(mode == 3):
    linearReg(iris_df)


