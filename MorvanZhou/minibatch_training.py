# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:40:19 2018

@author: yl
"""

import torch
import torch.utils.data as Data#进行小批训练的模块
torch.manual_seed(1)     # reproducible
'''
BATCH_SIZE=5
x=torch.linspace(1,10,10)   #this is x data (torch tensor)
y=torch.linspace(10,1,10)    # this is y data (torch tensor)

#torch_dataset=Data.TensorDataset(data_tensor=x,target_tensor=y)

torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,)


for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        #training...
        print('epoch',epoch,'| step:',step, '| batch x:', 
              batch_x.numpy(), '|batch y:', batch_y.numpy())
'''    
MINIBATCH_SIZE = 5    # mini batch size
x = torch.linspace(1, 10, 10)  # torch tensor
y = torch.linspace(10, 1, 10)

# first transform the data to dataset can be processed by torch
torch_dataset = Data.TensorDataset(x, y)
# put the dataset into DataLoader
loader =  Data.DataLoader(
    dataset=torch_dataset,
    batch_size=MINIBATCH_SIZE,
    shuffle=True,
    num_workers=2           # set multi-work num read data
)

for epoch in range(3):
    # 1 epoch go the whole data
    for step, (batch_x, batch_y) in enumerate(loader):
        # here to train your model
        print('\n\n epoch: ', epoch, '| step: ', step, '| batch x: ', batch_x.numpy(), '| batch_y: ', batch_y.numpy())
