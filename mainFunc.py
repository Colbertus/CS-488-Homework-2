# Needed imports 
from sklearn.datasets import load_iris
#from ucimlrepo import fetch_ucirepo
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

def heatMap():
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

def lin_reg(x, y):

    n = np.size(x)

    m_x, m_y = np.mean(x), np.mean(y)

    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    beta = SS_xy / SS_xx
    alpha = m_y - beta * m_x

    return(alpha, beta)

def plot_lin_reg_model(x, y, alpha, beta):
    plt.scatter(x, y, color = "m", marker = "o", s = 30)

    y_pred = alpha + beta * x

    plt.plot(x, y_pred, color = "g")

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def linearReg():
   

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
    linearReg()


