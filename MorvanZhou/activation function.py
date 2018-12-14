# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:58:39 2018

@author: yl
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

#fake data
x=torch.linspace(-5,5,200)#x data (tensor),shape=(100,1)
x=Variable(x)
x_np=x.data.numpy()#画图时tensor数据无法识别
y_relu=F.relu(x).data.numpy()
y_sigmoid=F.sigmoid(x).data.numpy()
y_tanh=F.tanh(x).data.numpy()
y_softplus=F.softplus(x).data.numpy()
#y_softmax=F.softmax(x)

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np,y_sigmoid,c='red',label='sigmoid')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')


plt.subplot(223)
plt.plot(x_np,y_tanh,c='red',label='tanh')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np,y_tanh,c='red',label='tanh')
plt.ylim((-0.2,6))
plt.legend(loc='best')