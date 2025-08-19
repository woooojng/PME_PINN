#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from Conditions import InitialCondition_u

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Set the numbers for each component of the net
        num_inputs = 2
        N = 128 #depth
        self.width = 6 #width

        self.hidden_layer_u1 = nn.ModuleList() #[]
        
        for i in range(self.width):
            if i ==0:
                self.hidden_layer_u1.append(nn.Linear(num_inputs, N))
                
                torch.nn.init.xavier_uniform_(self.hidden_layer_u1[i].weight)
            else:
                self.hidden_layer_u1.append(nn.Linear(N, N))
                torch.nn.init.xavier_uniform_(self.hidden_layer_u1[i].weight)
            
        self.hidden_layer_u1.append(nn.Linear(N,1)) 
        torch.nn.init.xavier_uniform_(self.hidden_layer_u1[-1].weight)

        self.optimizer = torch.optim.Adam(self.parameters()) #LBFGS or Adam
        
        self.x1_l = -1
        self.x1_u = 1

    def forward(self, x, t):
        bunch_input = torch.cat([x,t],axis=1)
        #flatten = nn.Flatten()
        #bunch_input = flatten(bunch_input)
        
        activation_function = torch.tanh

        #Pass the inputs through the layers in the above initiating setting.
        for i in range(self.width):
            if i ==0:
                out = activation_function(self.hidden_layer_u1[i](bunch_input))
            else:
                out = activation_function(self.hidden_layer_u1[i](out))
        output_u1 = self.hidden_layer_u1[-1](out)

        return output_u1

    def scaler(net, tensor):
        low = torch.min(tensor)
        high = torch.max(tensor)
        return (tensor-low)/(high - low)








