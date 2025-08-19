# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:52:56 2024

@author: kevbuck
"""

#### Benchmark II: 2D Cahn Hilliard, Seven Circles

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:05:35 2023

@author: kevbuck
"""

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class AC1Dnet2(nn.Module):
    #1 layer N node Neural Network
    def __init__(self, phi_layers):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.mse_cost_function = torch.nn.MSELoss()
        
        self.layers_phi = nn.ModuleList()
        
        #Phi Network
        self.layers_phi.append(nn.Linear(phi_layers[0], phi_layers[1]))
        torch.nn.init.xavier_uniform_(self.layers_phi[0].weight)
        
        for i in range(1, len(phi_layers) - 2):
            self.layers_phi.append(nn.Linear(phi_layers[i], phi_layers[i+1]))
            torch.nn.init.xavier_uniform_(self.layers_phi[i].weight)
        
        self.layers_phi.append(nn.Linear(phi_layers[-2], phi_layers[-1]))
        torch.nn.init.xavier_uniform_(self.layers_phi[-1].weight)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.optimizer_2 = torch.optim.LBFGS(self.parameters())
        
        ## Model Paramters defined here
        self.gamma_1 = 1e-4
        self.gamma_2 = 4
        
        self.x1_l = -1
        self.x1_u = 1
        self.t0 = 0
        self.tf = 1
    
    def forward(self, x, t):
       # relu = torch.nn.ReLU()
        y = torch.cat([x, t],axis=1)
        y = self.flatten(y)
        
        phi = y
        for i in range(len(self.layers_phi) - 1):
            temp = self.layers_phi[i](phi)
            phi = torch.tanh(temp)
        phi = self.layers_phi[-1](phi)
        
        return phi

    def W(self, x, t):
        phi = self(x, t)
        return .25*torch.pow((torch.pow(phi,2)-1), 2)
    
    def Energy(self, x, t):
        phi = self(x, t)
        
        W = .25*torch.pow((torch.pow(phi,2)-1), 2)
        
        phi_x = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        zero = torch.zeros_like(phi_x)
        
        E = self.gamma_2*W + self.gamma_1*self.mse_cost_function(phi_x, zero)
        
        return E
    
    def Energy_initial(self, x):
        phi = self.Initial_Condition(x)
        phi_x = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        zero = torch.zeros_like(phi_x)
        
        W = .25*torch.pow((torch.pow(phi,2)-1), 2)
        
        E = self.mse_cost_function(phi_x, zero) + W
        
        return E
    
    def W_exact(self, x):
        phi = self.Initial_Condition(x)
        return .25*torch.pow((torch.pow(phi,2)-1), 2)
    
    def Initial_Condition(self, x):
        
        phi_0 = torch.pow(x, 2) * torch.sin(2*np.pi*x)

        return phi_0
    
    def PDE_Loss(self, x, t):
        phi = self(x, t)
                    
        phi_x = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        phi_xx = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0]
        phi_t = torch.autograd.grad(phi.sum(), t, create_graph=True)[0]
        
        #compute loss
        pde = phi_t - self.gamma_1*phi_xx + self.gamma_2*torch.pow(phi, 3) - self.gamma_2*phi  #loss for allen cahn
        zeros = torch.zeros_like(pde)
        loss = self.mse_cost_function(pde, zeros)
        
        return loss
    
    def PDE_Loss_pointwise(self, x, t):
        phi = self(x, t)
                    
        phi_x = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        phi_xx = torch.autograd.grad(phi_x.sum(), x, create_graph=True)[0]
        phi_t = torch.autograd.grad(phi.sum(), t, create_graph=True)[0]
        
        #compute loss
        pde = phi_t - self.gamma_1*phi_xx + self.gamma_2*torch.pow(phi, 3) - self.gamma_2*phi  #loss for allen cahn

        loss = torch.pow(pde, 2)
        
        return loss
    
    def Initial_Condition_Loss(self, x):
        #IC loss for both phi and mu
        t = Variable(torch.zeros_like(x), requires_grad=True).to(device)
        
        phi_pred = self(x, t)
        phi_exact = self.Initial_Condition(x)
        zero = torch.zeros_like(phi_exact)

        initial_condition_loss = self.mse_cost_function(phi_pred, phi_exact)/self.mse_cost_function(phi_exact, zero)
        
        return initial_condition_loss

    def Initial_Condition_Loss_pointwise(self, x):
        #IC loss for both phi and mu
        t = Variable(torch.zeros_like(x), requires_grad=True).to(device)
        
        phi_pred = self(x, t)
        phi_exact = self.Initial_Condition(x)

        initial_condition_loss = torch.pow(phi_pred-phi_exact, 2)
        
        return initial_condition_loss
    
    def Boundary_Loss(self, t):
        
        x1_l_pt = Variable(self.x1_l * torch.ones_like(t), requires_grad=True).to(device)
        x1_u_pt = Variable(self.x1_u * torch.ones_like(t), requires_grad=True).to(device)
        
        #Evaluate at the 4 boundaries
        phi_left = self(x1_l_pt, t)
        phi_right = self(x1_u_pt, t)
        
        #homogeneous neumann on phi and mu
        phi_left_x1 = torch.autograd.grad(phi_left.sum(), x1_l_pt, create_graph=True)[0]
        phi_right_x1 = torch.autograd.grad(phi_right.sum(), x1_u_pt, create_graph=True)[0]
        
        #Periodic Boundary
        phi_dirichlet = self.mse_cost_function(phi_left, phi_right)
        phi_neumann = self.mse_cost_function(phi_left_x1, phi_right_x1)
        
        #total loss
        boundary_loss = phi_dirichlet + phi_neumann
                
        return boundary_loss
    