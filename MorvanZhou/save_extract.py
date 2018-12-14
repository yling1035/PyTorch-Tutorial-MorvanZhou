# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:08:56 2018

@author: yl
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)  #reproducible  
# fake data
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)  #x data (tensor),shape=(100,1)
y=x.pow(2)+0.2*torch.rand(x.size())  #noisy y data (tensor),shape=(100,1)
x,y=Variable(x,requires_grad=False),Variable(y,requires_grad=False)


def save():
    net1=torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
    )
    optimizer=torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func=torch.nn.MSELoss()
    for i in range(100):
        prediction=net1(x)
        loss=loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    torch.save(net1,'net.pkl')  # save entire net    
    torch.save(net1.state_dict(),'net_params.pkl')  # save parameters
    
    #plot result
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)

def restore_net():  #extract net
    net2=torch.load('net.pkl')
    prediction=net2(x)
    #plot result
    plt.figure(1,figsize=(10,3))
    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
    

def restore_params():  #extract parameters
    net3=torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction=net3(x)
    #plot result
    plt.figure(1,figsize=(10,3))
    plt.subplot(133)
    plt.title('net3')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)