# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:55:18 2020

@author: taylo
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\01-linear_regression')
# have data for proﬁts and populations from the cities, then plot
data = pd.read_csv('ex1data1.txt',names=['Population (x_1)','Profit (Y)'])
data.plot.scatter('Population (x_1)','Profit (Y)',c='red',marker='+',
                 label='profit of population')
#plt.show()
data.insert(0,'x_0',1) #insert x_0 column
cols=data.shape[1]  #extract number of column
print('There are '+str(cols)+' columns')
x=data.iloc[:,0:cols-1] #define x
y=data.iloc[:,cols-1:cols] #define y

print(x.shape) #check x dimensions,(97, 2)
print(y.shape) #check y dimensions,(97, 1)

X=np.matrix(x.values) #以列表形式返回字典中的所有值,然后转化为 numpy 矩阵
Y=np.matrix(y.values)
theta=np.zeros((2,1)) #Xθ 的维度等于 y 的维度，计算得到 θ 的维度为（2，1）
#define cost function
def costFunction(X,Y,theta):
    inner=np.power(X*theta-Y,2)
    return np.sum(inner)/(2*len(X))

#implement gradient descent
def gradientDescent(X,Y,theta,alpha, iters):    # iters为梯度下降中的迭代次数
    costs=[]    # 将每次迭代的代价函数值保存在列表
    for i in range(iters):
        theta = theta-(X.T*(X*theta-Y))*alpha/len(X)
        cost=costFunction(X,Y,theta)
        costs.append(cost)
    if i%100==0:
        print(cost)
    return theta,costs #只返回最后一个theta,因为最后一个最小


alpha=0.02
iters=1000
theta,costs=gradientDescent(X,Y,theta,alpha,iters) #返回迭代1000次后

#plot iters and cost
fig,ax1=plt.subplots()
ax1.plot([i for i in range(iters)],costs)
ax1.set_xlabel('iters')
ax1.set_ylabel('cost')
plt.show()

#visualization
fig,ax = plt.subplots()
x1=np.linspace(X.min(),X.max(),100)
y1=theta[0,0] + (theta[1,0]*x1)
ax.plot(x1,y1,'b',label='Prediction')
ax.scatter(X[:,1].tolist(),Y.tolist(),label='Training data',color='r',marker='+')
ax.legend()
ax.set(xlabel='Population of City in 10,000s',ylabel='Profit in $10,000s',title='Figure 2: Training data with linear regression fit')
plot.show()
