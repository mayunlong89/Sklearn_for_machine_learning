# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:05:25 2020

@author: Yunlong Ma
"""

#---------------------Start-----------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

plt.figure
X,Y = make_classification(n_samples = 300, n_features =2, n_redundant=0, n_informative=2,random_state=22,n_clusters_per_class=1,scale=100)

plt.scatter(X[:,0],X[:,1],c=Y)

plt.show()


#using minmax method to normalization
X=preprocessing.minmax_scale(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)
clf =SVC()
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))




#------------------End--------------------------------------------------------

