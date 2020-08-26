# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:58:50 2020

@author: Yunlong Ma
@E-mail: glb-biotech@zju.edu.cn
"""

#-------------------------Start--------------------------------------------------------------

#Model_1 
#import datasets package
# use train_test_split method to split data into training and testing datasets
from sklearn import datasets 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier #use the method to train data
from sklearn.model_selection import cross_val_score #used for 5-fold or 10-fold cross-validation

#import data
iris = datasets.load_iris() 
iris_X = iris.data #charateristics variable
iris_Y = iris.target #target value

X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size = 0.3)

print (Y_train)

#training
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)


#predicting data
print(knn.predict(X_test))
pred_value = knn.predict(X_test)

#real data
print(Y_test)
real_value = Y_test

#function for calculating the accuracy
def accuracy_calc(x,y):
    length = len(x)
    count = 0
    for i in range(length):
        if x[i] == y[i]:
            count = count + 1
    accuracy = count/length
    return "Predicting accuracy: ", accuracy


if __name__ == '__main__':
    value = accuracy_calc(pred_value, real_value)
    print(value[0], '%.3f'%value[1])
    #add a 5-fold cross-validation
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn,iris_X,iris_Y, cv=5, scoring = "accuracy") #use "accuracy" method as a scoring method
    print("score", scores)  #cross-validation score
    print('score.mean',scores.mean()) #mean for cross-validation scores



#----------------------------End-----------------------------------------------------






