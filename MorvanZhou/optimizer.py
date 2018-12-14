# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:59:21 2018

@author: yl
"""

import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# hyper patameters
LR=0.01
BATCH_SIZE=32
EPOCH=12

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)# unsqueeze将1维变为2维
y=x.pow(2)+0.2*torch.rand(x.size())

# plot dataset
plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

torch_dataset=Data.TensorDataset(data_tensor=x,targer_tensor=y)
loader=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_works=2)

class Net(torch.nn.Module):
    def _init_(self,n_features,n_hidden,n_output):
        super(Net,self)._init_()#继承Net模块到神经网络模块
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
        
    
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x
 
# different nets
net_SGD=     Net()
net_Momentum=Net()
net_RMSprop= Net()
net_Adam=    Net()
nets= [net_SGD,net_Momentum,net_RMSprop,net_Adam]

#optimizer
opt_SGD     =torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop =torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam    =torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

loss_func=torch.nn.MSELoss()
losses_his=[[],[],[],[]]# record loss

for epoch in range(EPOCH):
    print(epoch)
    for step,(batch_x,batch_y) in enumerate(loader):
        b_x=Variable(batch_x)
        b_y=Variable(batch_y)
        for net,opt,l_his in zip(nets,optimizers,losses_his):
            output=net(b_x)
            loss=loss_func(output,b_y)
            opt.zero_grad()  #clear gradients for next train
            loss.backward()  #backpropagation,compute gradients
            opt.step()  #apply gradients
            l_his.append(loss.data[0]) #loss recorder
labels=['SGD','Momentum','RMSprop','Adam']
for i,l_his in enumerate(losses_his):
    plt.plot(l_his,label=labels[i]) 
plt.legend(loc='best')
plt.xlabel('step')
plt.ylabel('loss')
plt.ylim((0,0.2))
plt.show()
           
        
    