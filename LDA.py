# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inputfile='xigua3.0.xlsx'

data = pd.read_excel(inputfile)
x = data.values[:,0:2]
y = data.values[:,2]
dataset0 = np.array([list(data[u'密度']),list(data[u'含糖率']),list(data[u'好瓜'])])
dataset0 = dataset0.T
dataset = dataset0[:, 0: 2]
is_good = dataset0[:, 2]
u = []
for i in range(2):
    u.append(np.mean(x[y == i],axis = 0)) 
m,n = np.shape(x)
sw = np.zeros((n,n))
for i in range(m):
    x_tmp = x[i].reshape(n,1)  #将x的每行数据转为n行1列
    if y[i] == 0:
        u_tmp = u[0].reshape(n, 1)
    if y[i] == 1:
        u_tmp = u[1].reshape(n, 1)
    sw += np.dot(x_tmp - u_tmp,(x_tmp - u_tmp).T)
sw = np.mat(sw) #转为矩阵
u,sigma,v = np.linalg.svd(sw)
sw_inv = v.T * np.linalg.inv(np.diag(sigma)) * u.T
w = np.dot(sw_inv, (u[0] - u[1]).reshape(n, 1))
f = plt.figure(1)
D = np.linspace(0.2, 0.8, 1000)
a = w.A
print(a)
print(a[0])
S = -1 * (D * a[0]) * (1 / a[1])
plt.plot(D, S)
#print(w[0])
#print(w[1])
plt.title(u'watermelon_dataset_3.0α')
plt.xlabel(u'density')
plt.ylabel(u'sugar_ratio')
plt.scatter(dataset[is_good == 1, 0], dataset[is_good == 1, 1], marker = 'o', color = 'r', s = 100, label = 'good')
plt.scatter(dataset[is_good == 0, 0], dataset[is_good == 0, 1], marker = 'o', color = 'g', s = 100, label = 'bad')
plt.legend(loc = 'upper right')        
plt.show()

if __name__ == '__main__':
    print(w)
