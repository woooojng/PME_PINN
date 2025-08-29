#!/usr/bin/env python
# coding: utf-8

import torch
from torch.autograd import Variable
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lossIC(net, x):
    mse_cost_function = torch.nn.MSELoss()
    
    #Compute estimated initial condition
    zero = torch.zeros_like(x).to(device)
    
    u = net(x, zero) 
    
    #Compute actual initial condition
    u_0 = InitialCondition_u(net, x)
    
    u_IC_loss = mse_cost_function(u, u_0)
    u_IC_loss_scaler = mse_cost_function(u_0, zero)
     
    return u_IC_loss/u_IC_loss_scaler 
    
def parabola_peak(center, width, height, xgrid):
    zero = torch.zeros_like(xgrid).to(device)
    return height * torch.maximum(zero, 1 - ((xgrid - center) / width) ** 2)
    
def InitialCondition_u(net, x): 
    zero = torch.zeros_like(x).to(device)
    #n =net.n 
    #m = net.m
    #alpha = n/(n*(m-1)+2)
    #beta = 1/(n*(m-1)+2)
    #t0 = .01

    #C = .2
    #K = (m-1)/(2*m) *beta #.01 #((m-1)/(2m) * beta
    u_0 = ( 
        parabola_peak(.3, 0.15, 0.9, x)    # left wide peak
        + parabola_peak(-0.3, 0.35, 0.4, x)
    )
    
    return u_0

def lossBdry(net, t):
    mse_cost_function = torch.nn.MSELoss()
    zero = torch.zeros_like(t).to(device)
    
    x_l_Bdry = Variable(net.x1_l * torch.ones_like(t), requires_grad=True).to(device)
    x_u_Bdry = Variable(net.x1_u * torch.ones_like(t), requires_grad=True).to(device)
    
    
    # Define the variables on 4 boundaries at outer squre
    
    u_left = net(x_l_Bdry, t)
    u_right = net(x_u_Bdry, t)
    
    u_x_left = torch.autograd.grad(u_left.sum(), x_l_Bdry, create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right.sum(), x_u_Bdry, create_graph=True)[0]
    
    
    loss_u = mse_cost_function(u_left, u_right)
    loss_u_x = mse_cost_function(u_x_left, u_x_right)
    
    return loss_u, loss_u_x

def lossNSpde(net, x, t):
    mse_cost_function = torch.nn.MSELoss()
    m = net.m
    u = net(x, t)
    zero = torch.zeros_like(x).to(device)
    
    #Compute Derivatives
    
    h_x = torch.autograd.grad(u**m, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    h_xx = torch.autograd.grad(h_x, x, grad_outputs=torch.ones_like(h_x), create_graph=True)[0]

    #Loss functions w.r.t. governing Navier-Stokes equation on inner space
    #AC
    AC_Residual = (u_t - h_xx) #*torch.log(np.exp(1)+(np.exp(5) - np.exp(1))*t)
    loss = mse_cost_function(AC_Residual, zero)
    raw_loss = mse_cost_function((u_t - h_xx), zero)
    return loss, raw_loss

def lossNSpde_rank(net, x, t,batch_size):
    mse_cost_function = torch.nn.MSELoss()
    
    u = net(x, t)
    m = net.m
    
    #Compute Derivatives
    
    h_x = torch.autograd.grad(u**m, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    h_xx = torch.autograd.grad(h_x, x, grad_outputs=torch.ones_like(h_x), create_graph=True)[0]

    #Loss functions w.r.t. governing Navier-Stokes equation on inner space
    #AC
    AC_Residual = (u_t - h_xx) #*torch.log(np.exp(1)+(np.exp(5) - np.exp(1))*t)
    
    #loss = mse_cost_function(AC_Residual, zero)
    loss = torch.abs(AC_Residual)
    loss.sort(descending=True)
    loss = loss.reshape(-1,1)
    #sorted_tensor, indices = torch.sort(torch.reshape(loss, (-1,1)), descending=True) #PDEloss_tensor.view(-1)
    #sorted_tensor = torch.reshape(sorted_tensor, (-1,1) )#sorted_tensor.view(-1, 1)
    
    max_stad = loss[int(batch_size/10-1)][0]
    PDEloss_picked = torch.where(loss>=max_stad, loss, 0)
    zero = torch.zeros_like(PDEloss_picked).to(device)
    
    loss = mse_cost_function(PDEloss_picked, zero) #adjust denominator from len(x[:,0]) to len(x[:,0])/10
    raw_loss = mse_cost_function((u_t - h_xx), zero)
    return loss, PDEloss_picked, raw_loss

def total_free_e_derivative(net, x, t):
    mse_cost_function = torch.nn.MSELoss()
    zero = torch.zeros_like(x).to(device)
    if t == None:
        u = InitialCondition_u(net, x)
        
    else:
        u = net(x, t)
        

    #e_derivative = (u**net.m)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    #e_x = net.scaler(torch.abs(e_derivative))
    '''
    r = 
    t_l = 0
    t_u = 1
    exp = - torch.log(1-e_x+e_x*np.exp(-r*(t_u-t_l)))/r
    '''
    return mse_cost_function(u**(net.m-1) * u_x, zero) + 1/(net.m-1)* torch.abs(u**(net.m))/(net.m)/(net.m)
    
