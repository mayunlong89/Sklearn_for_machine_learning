# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:05:25 2020

"""

#--------------Start------------------------------------
#Model_2 form our own datasets
    
from sklearn import datasets
from sklearn.linear_model import LinearRegression

load_data = datasets.load_boston()
data_X = load_data.data
data_Y = load_data.target
print(data_X.shape)


model = LinearRegression()
model.fit(data_X,data_Y)
model.predict(data_X[:4,:])

print(model.coef_)

print(model.intercept_)
print(model.get_params())
print(model.score(data_X, data_Y))
print(LinearRegression.score(data_X, data_Y))



#---------------End-----------------------------------


