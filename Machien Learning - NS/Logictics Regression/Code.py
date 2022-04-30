import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%

dataframe = pd.read_csv("data_classification.csv",header = None)
dataframe.head()
#%%

true_x = []
true_y = []
false_x = []
false_y = []
for i in dataframe.values:
    if i[2] == 1 :
        true_x.append(i[0])
        true_y.append(i[1])
    else:
        false_x.append(i[0])
        false_y.append(i[1])

plt.scatter(true_x,true_y, marker='o')
plt.scatter(false_x,false_y, marker='x')
#%%

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def classify(p):
    if p > 0.5:
        return 1
    else:
        return 0

def predict(feature,weight):
    z = np.dot(feature,weight)
    return sigmoid(z)

def cost_function(feature,labels,weight):
    '''
    :param feature:
    :param label:
    :param weight:
    :return:
    '''
    n = len(labels)
    predictions = predict(feature,weight)
    cost_class1 = -labels * np.log(predictions)
    cost_class2 = -labels * np.log(1 - predictions)
    cost = cost_class1 + cost_class2
    return cost.sum()/n

def gradient(feature,labels,weight,learning_rate):
    '''
    :param feature:
    :param labels:
    :param weight:
    :param learning_rate: float
    :return: new weight
    '''
    n = len(labels)
    predictions = predict(feature,weight)
    gd = np.dot(feature.T,predictions - labels)
    gd = gd/n
    gd = gd*learning_rate
    weight = weight-gd
    return  weight

def train(feature, labels, weight, learning_rate,iter):
    cost_hs = []
    for i in range(iter):
        weight = gradient(feature,labels,weight,learning_rate)
        cost = cost_function(feature,labels,weight)
        cost_hs.append(cost)

    return weight, cost_hs



#%%
