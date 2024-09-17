# Implementation of Logistic Regression Using Gradient Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.

## Program:

Program to implement the the Logistic Regression Using Gradient Descent.

Developed by : KEERTHIVASAN S

RegisterNumber : 212223220046

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:

## Array Value of x : -

![arrayvalueofx](https://github.com/user-attachments/assets/151f6381-9b61-491f-81c7-3e3945ea0b1e)

## Array Value of y : -

![arrayvalueofy](https://github.com/user-attachments/assets/85b56c96-d346-4bee-b6e4-5a40b6b46691)

## Exam 1 - Score Graph : -

![exam1score](https://github.com/user-attachments/assets/e2807fa2-6d09-4395-ba99-be7d65980bc3)

## Sigmoid function graph : -

![sigmodfunctiongraph](https://github.com/user-attachments/assets/4b969238-bd0d-4640-94bf-5a5e3e3490ae)

## X train grad value : -

![xtraingradgraph](https://github.com/user-attachments/assets/de2244f4-414a-4b4f-bf79-61a58d3dd44e)

## Y train grad value : -

![ytraingradgraph](https://github.com/user-attachments/assets/1ce36482-e206-4bd9-b26e-7b9da5d1e524)

## print res.x : -

![res x](https://github.com/user-attachments/assets/42c32c1a-194a-4edd-b8a7-c386dd51e0f7)

## Decision Boundary - Graph for Exam Score : -

![decisionboundary](https://github.com/user-attachments/assets/ae215975-6271-4aab-a87f-b2e67b8f3559)

## Probability Value : -

![probabilityvalue](https://github.com/user-attachments/assets/2603edd7-9fb3-473e-91fe-81a824c309fe)

## Prediction value of Mean : -

![predictionvalueofmean](https://github.com/user-attachments/assets/1139c85e-5a74-4d17-b5fb-7b86e1ddf6ba)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
