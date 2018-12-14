# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:39:55 2018

@author: yl
"""
import torch 
import numpy as np

np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)
tensor2array=torch_data.numpy()
print '\nnumpy',np_data,'\ntorch',torch_data,'\ntensor2array',tensor2array

#abs
data=[-1,-2,1,2]
tensor=torch.FloatTensor(data)

data1=[[1,2][3,4]]
data=np.array(data1)
print '\nnumpy:',data1.dot(data1),'\ntorch:',tensor.dot(tensor)
