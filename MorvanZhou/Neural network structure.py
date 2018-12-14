# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:58:18 2018

@author: yl
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:17:16 2018

@author: yl
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)# unsqueeze将1维变为2维
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x),Variable(y)

plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

# method 1
class Net(torch.nn.Module):
    def _init_(self,n_features,n_hidden,n_output):
        super(Net,self)._init_()#继承Net模块到神经网络模块
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
        
    
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x
net1=Net(2,10,2)
print(net1)

# methohd 2
net2=torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,2),
        )
print(net2)

plt.ion() # something about plotting 
plt.show()

optimizer=torch.optim.SGD(net1.parameters(),lr=0.5)
loss_func=torch.nn.MSELoss()

for i in range(100):
    prediction=net1(x)
    loss=loss_func(prediction,y)
    
    optimizer.zero_grad()#进行参数更新时梯度降为0
    loss.backward()#反向传播
    optimizer.step()
    if i%5==0:
        #plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'loss=%.4f'%loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
        
plt.ioff()
plt.show()
    