# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:34:16 2020

@author: taylo
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\01-linear_regression')
data=pd.read_csv('ex1data2.txt', names=['size of the house','number of bedrooms','price of the house'])

#Feature Normalization
def featureNormalization(data):
    return (data-data.mean())/data.std()
data=featureNormalization(data)

#define variable
data.insert(0,'X0',1)
col=data.shape[1]
print('There are '+str(col)+' columns')
X=data.iloc[:,0:col-1]
Y=data.iloc[:,col-1:col]
X=np.matrix(X.values)
Y=np.matrix(Y.values)

#plot dataset
#ax1= plt.figure().add_subplot(111, projection = '3d')
#ax1.scatter3D(data.iloc[:,col-3:col-2],data.iloc[:,col-2:col-1],data.iloc[:,col-1:col], cmap='Blues',label='Training data')
#ax1.set(xlabel='size', ylabel='number of bedrooms',zlabel='price')
#plt.show()

#define theta
print(X.shape)
print(Y.shape)
theta=np.zeros((X.shape[1],Y.shape[1]))
print(theta.shape)

#define cost function
def costFunction(X,Y,theta):
    inner=np.power(X*theta-Y,2)
    return np.sum(inner)/(2*len(X))

#define gradient desent
def gradientDescent(X,Y,theta,alpha,iters):
    costs=[]
    for i in range(iters):
        theta = theta-(X.T*(X*theta-Y))*alpha/len(X)
        cost=costFunction(X,Y,theta)
        costs.append(cost)
    return theta, costs

#set alpha, and plot cost function through different alpha
candidate_alpha = [0.0003, 0.003, 0.03, 0.0001, 0.001, 0.01]
iters=2000
fig,ax=plt.subplots()
for alpha in candidate_alpha:
    _, costs = gradientDescent(X,Y,theta,alpha,iters) #cannot use 'theta' as variable! there are alpha loops, if set'theta' as variable, the former theta result will be as used as the next loop's starting theta rather the initialized value 
    ax.plot([i for i in range(iters)],costs,label=alpha)
    ax.legend()
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Cost')
plt.show()

#set alpha to be equal to 0.003
alpha_1=0.003
theta1,costs= gradientDescent(X,Y,theta,alpha_1,iters)
#plot dataset and the prediction model
x1=np.linspace(X[:,1].min(),X[:,1].max(),100) #生成的样本数据量为100
y1=np.linspace(X[:,2].min(),X[:,2].min(),100)
#x1,y1=np.meshgrid(np.linspace(X[:,1].min(),X[:,1].max(),100),np.linspace(X[:,2].min(),X[:,2].min(),100))
z1=theta1[0,0]+(theta1[1,0]*x1)+(theta1[2,0]*y1)

print(z1)

#ax= plt.figure().add_subplot(111, projection = '3d')
#fig = plt.figure()
#ax = fig.gca(projection='3d')
# 创建画布
fig = plt.figure(figsize=(12, 8),facecolor='lightyellow')

# 创建 3D 坐标系
ax = fig.gca(fc='whitesmoke',projection='3d')
ax.scatter3D(data.iloc[:,col-3:col-2],data.iloc[:,col-2:col-1],data.iloc[:,col-1:col], cmap='Blues',label='Training data')
ax.plot3D(x1,y1,z1,'green',label='Prediction')
ax.set(xlabel='size', ylabel='number of bedrooms',zlabel='price')
plt.legend()
plt.show()
