# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:17:16 2018

@author: yl
"""

import torch
from torch.autograd import Variable# 用来包住数据
import torch.nn.functional as F
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)# unsqueeze将1维变为2维，因为torch只会处理二维数据
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x),Variable(y)

plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

class Net(torch.nn.Module):#里面有神经网络的模块
    def _init_(self,n_features,n_hidden,n_output):#定义
        super(Net,self)._init_()#继承Net模块到神经网络模块
        self.hidden=torch.nn.Linear(n_features,n_hidden)#层
        self.predict=torch.nn.Linear(n_hidden,n_output)
        
        #搭建层
    
    def forward(self,x):
        x=F.relu(self.hidden(x))#激活函数
        x=self.predict(x)# 负无穷到正无穷
        return x
net=Net(1,10,1)#一个输入，10个神经元，1个输出
print(net)

plt.ion() # something about plotting 
plt.show()
#神经网络的优化
optimizer=torch.optim.SGD(net.parameters(),lr=0.5)
loss_func=torch.nn.MSELoss()#计算误差的函数，均方差

for i in range(100):
    prediction=net(x)
    loss=loss_func(prediction,y)
    
    optimizer.zero_grad()#优化参数，每次进行参数更新时梯度必须要将降为0
    loss.backward()#反向传播
    optimizer.step()#优化梯度
    if i%5==0:
        #plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'loss=%.4f'%loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1) #暂停0.1s
        
plt.ioff()
plt.show()
    