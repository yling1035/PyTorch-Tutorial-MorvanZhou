# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 19:07:11 2018

@author: yl
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision#数据库
import matplotlib.pyplot as plt


#Hyper
EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False

train_data=torchvison.datasets.MNIST(root='./mnist/',
                                     train=True,
                                     transform=torchvision.transforms.ToTensor(),#（0-255）压缩到(0,1)
                                     download=DOWNLOAD_MNIST
                                     )
#plot one example
print(train_data.train_data.size())    #(6000,28,28)
print(train_data.train_labels.size()) #6000
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')#呈现第一张照片
plt.title('%i' % train_data.train.labels[0])
plt.show()

train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
test_data=torchvision.datasets.MNIST(root='./mnist/',train=False)

test_x=Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255
test_y=test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(   #(1,28,28,)
               nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=2,padding=2),#(16,28,28)
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2),#(16,14,14)
               )
        self.conv2=nn.Sequential(
               nn.Conv2d(16,32,5,1,2),#(16,14,14)
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2),#(32,7,7) )
               )
        self.out=nn.Linear(32*7*7,10)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)  #(batch,32,7,7)
        x=x.view(x.size(0),-1)  #(batch,32*7*7)
        output=self.out(x)
        return output

cnn=CNN()
print(cnn)
    
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss() 

#training and testing
for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x=Variable(x)
        b_y=Variable(y)
        
        output=cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50==0:
            test_output=cnn(test_x)
            pred_y=torch.max(test_output,1)[1].data.squeeze()
            accuracy=sum(pred_y==test_y)/test_y.szie(0)
            print('epoch:', epoch, '|train loss:%.4f' % loss.data[0], '|test accuracy:5.4f' % accuracy)
            
 # print 10 predictions from test data
test_output=cnn(test_x[:10])
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')
            
            
        
        
        
        
    
    