# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:44:01 2018

@author: yl
"""
import torch
from torch.autograd import Variable
tensor=torch.FloatTensor([[1,2],[3,4]])
variable=Variable(tensor,requires_grad=True)
print tensor
print variable

t_out=torch.mean(tensor*tensor)
v_out=torch.mean(variable*variable)
print t_out
print v_out


#tensor不能进行反向传播，variable能进行反向传播
v_out.backward()
print variable.grad #梯度更新
print variable#变量
print variable.data #tensor的形式
print variable.data.numpy() #data 是tensor的形式，将转换为numpy