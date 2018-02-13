
#Multiple Linear Regresssion

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sklearn.preprocessing.imputation as imp
#import sklearn.preprocessing.label as lab

#Import dataset using Pandas
dataset = pd.read_csv('50_Startups.csv') #placed project py file in same directory as Data.csv to avoid fully qualified directory path


x=dataset.iloc[:, :-1].values #independent variables - given data columns
y=dataset.iloc[:, 4].values  #dependend variabls-predicted data column

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap
x = x[:, 1:]

#Splitting training and test data sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0) #split 20% test/80% train


#Fitting Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(x_train, y_train) 

#Predicting the Test set results
y_pred = regressor.predict(x_test)


#backward elimination
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50,1)).astype(int),values=x, axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=x_opt).fit() #fit with all possible predictors
regressor_OLS.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=x_opt).fit() #fit with all possible predictors
regressor_OLS.summary()
