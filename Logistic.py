# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt

inputfile = 'xigua3.0.xlsx'

data = pd.read_excel(inputfile, 'Sheet1')
dataset0 = np.array([list(data[u'密度']),list(data[u'含糖率']),list(data[u'好瓜'])])
dataset0 = dataset0.T
dataset = dataset0[:, 0: 2]
is_good = dataset0[:, 2]
f = plt.figure(1)
plt.title(u'watermelon_dataset_3.0α')
plt.xlabel(u'density')
plt.ylabel(u'sugar_ratio')
x = np.array([list(data[u'密度']),list(data[u'含糖率']),[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
x = x.T
y = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
beta = np.array([[rand.uniform(-1, 1)],[rand.uniform(-1, 1)],[rand.uniform(-1, 1)]])
l_beta = 0
old_l_beta = 0
n = 0
while True:
    beta_T = np.transpose(beta)
    for i in np.arange(len(x)):
        l_beta = l_beta + (-y[i]*np.dot(beta_T, np.array([x[i,:]]).T) +  np.log(1+np.exp(np.dot(beta_T,np.array([x[i,:]]).T))))
    if n >= 5000:
        D = np.linspace(0.2, 0.8, 1000)
        S = -1 * (D * beta[0] + beta[2]) * (1 / beta[1])
        plt.plot(D, S)
        plt.title(u'watermelon_dataset_3.0α')
        plt.xlabel(u'density')
        plt.ylabel(u'sugar_ratio')
        plt.scatter(dataset[is_good == 1, 0], dataset[is_good == 1, 1], marker = 'o', color = 'r', s = 100, label = 'good')
        plt.scatter(dataset[is_good == 0, 0], dataset[is_good == 0, 1], marker = 'o', color = 'g', s = 100, label = 'bad')
        plt.legend(loc = 'upper right')        
        plt.show()
        break
    dbeta = 0
    d2beta = 0
    n = n+1
    old_l_beta = l_beta
    for i in np.arange(len(x)):
        x_i = np.array([x[i,:]])
        x_i_2 = np.dot(x_i,x_i.T)
        exp_b_x = np.exp(np.dot(np.transpose(beta),x_i.T))  
        dbeta = dbeta - np.array([x[i,:]])*(y[i]-(exp_b_x/(1+exp_b_x)))
        d2beta = d2beta + x_i_2*exp_b_x/((1+exp_b_x)*(1+exp_b_x))
    beta = beta - np.dot(np.linalg.inv(d2beta),dbeta).T

if __name__ == '__main__':
    print(beta)
