# Author Alec Reid
# Sources https://www.kaggle.com/code/manjunathnp/linear-regression-mileage-prediction-for-mtcars

import os
os.chdir ("c:/AlecProjects/Machine-Learning_AR/week_1_exercise")
print(os.getcwd())

#Imports
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
import numpy as np

#Read in CSV
dfcars=pd.read_csv("mtcars.csv")

dfcars=dfcars.rename(columns={"Unnamed: 0":"name"})
print (dfcars.head())
print (dfcars.shape)
#print (dfcars.info)

# Skewness means veering away from a symmetrical bell curve. It presents a curve that is asymmetrical and has a tail extending toward the right (positive skew) or to the left (negative skew)
print("Skewness of 'hp': ", stats.skew(dfcars.hp))
print("Skewness of 'wt': ", stats.skew(dfcars.wt))

# Measure of Kurtosis
print("Kurtosis of 'hp': ", stats.kurtosis(dfcars.hp))
print("Kurtosis of 'wt': ", stats.kurtosis(dfcars.wt))

# Visually respresenting this for 'hp'
sb.displot (dfcars.hp)
