#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import pymatgen as mg
import pymatgen.analysis.diffraction as anadi
import pymatgen.analysis.diffraction.xrd as xrd
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable


# In[ ]:


torch.set_default_dtype(torch.float64)

torch.set_printoptions(precision=8)

patt_xrd = xrd.XRDCalculator('CuKa')

global sample_num, rmat_num
sample_num=1
rmat_num=28  #row nums of the matrix for the input of CNN 

global move_num=-91.63007225
global extend_num=10000
#en_mean=-3.05380

LR_D=0.001  #learning rate
LR_G=0.001


# In[ ]:


def random_xxpsk(file_path):
    folder=np.random.choice(glob.glob(file_path +"*"))
    #pos_name=folder+'/POSCAR'
    #out_name=folder+'/OUTCAR'
    return folder

def tomgStructure(folder):
    POSfile=folder+'/POSCAR'      
    R_mgS=mg.Structure.from_file(POSfile)
    return R_mgS

###
##input_data_to_model:extract_PXRD_data_and_select_peak
###
def get_xrdmat3(mgStructure):
    global rmat_num
    xrd_data3 =patt_xrd.get_pattern(mgStructure)
    i_column = rmat_num
    xrow = len(xrd_data3.x)
    xrd_x= []
    xrd_y= []
    xxx=[]
    yyy=[]
    xrd_mat3=[]
    if xrow < i_column:
        for i in xrd_data3.x:
            xxx.append(i)
        for j in xrd_data3.y:
            yyy.append(j)
        for i in range(0,i_column-xrow):
            xxx.append(0)
            yyy.append(0)
        xrd_x=np.asarray(xxx)
        xrd_y=np.asarray(yyy)
    if xrow > i_column:
        xrd_x=xrd_data3.x[:i_column]
        xrd_y=xrd_data3.y[:i_column]
    if xrow == i_column:
        xrd_x= xrd_data3.x
        xrd_y= xrd_data3.y
    xrd_x=np.sin(np.dot(1/180*np.pi,xrd_x))
    xrd_y=np.dot(1/100,xrd_y)
    xrd_mat3.append(xrd_x)
    xrd_mat3.append(xrd_y)
    xrd_mat3=np.array(xrd_mat3)
    return xrd_mat3


##input_data_for_G:calculate_inner_product_i.e_L_matrix
###
def GANs_Gmat(Random_Structure):
    global rmat_num
    RS_xrdmat = get_xrdmat3(Random_Structure)
    multimat3_RS =  np.zeros((rmat_num,rmat_num),dtype='float32')
    multimat3_RS = np.asarray(np.sqrt(np.dot(RS_xrdmat.T, RS_xrdmat)))
    return multimat3_RS


###
##extract_input_porperty_data_by_DFT
###
def get_energy(folder):
    energy_string=os.popen('grep TOTEN '+folder+'/OUTCAR | tail -1').read().split(' ')[-2]
    energy=np.float64(float(energy_string))
    return energy_transform

def linear_transform(energy):
    global extend_num, move_num
    energy_transform=(energy-move_num)*extend_num
    return energy_transform

def inverse_transform(energy_transform):
    global entend_num, move_num
    energy=energy_transform/extend_num+move_num
    return energy

def get_atoms_num(folder2):
    xxx=tomgStructure(folder2)
    anum=len(xxx.sites)
    return anum

def get_energy_per_atom(energy_total,anum):
    energy_per_atom=energy_total/anum
    return energy_per_atom
##
###



###




# In[ ]:


class GNet(nn.Module):
    
    def __init__(self, input_size=(sample_num,28,28)):
        super(GNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(#(3,28,28)
                in_channels=sample_num,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#->(32,28,28)
            nn.ReLU(),#->(32,28,28)
            nn.MaxPool2d(kernel_size=2),
        )#->(#->(32,14,14))
        self.conv2=nn.Sequential(#->(32,14,14))
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#->(64,14,14)
            nn.ReLU(),#->(64,14,14)
            nn.MaxPool2d(kernel_size=2),#->(64,7,7)
        )
        self.out=nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,sample_num),            
        )
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x) #batch(64,7,7)
        x=x.view(x.size(0),-1) #(batch, 64*7*7)
        output=torch.unsqueeze(self.out(x),dim=0)
        return output

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.Dlstm=nn.LSTM(
            input_size=sample_num,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Sequential(
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid(),
        )
        #nn.Linear(32,1)
        #nn.Relu
        #nn.Linear
        #nn.Sigmoid
        
    def forward(self,x):
        D_out,(h_n,h_c)=self.Dlstm(x,None)
        out = self.out(D_out[:,-1,:]) #(batch,time step,input)   
        return out


# In[ ]:


G=GNet()
D=DNet()


# In[ ]:


G.load_state_dict(torch.load('./G_MATGANIP_energy.pkl'))
D.load_state_dict(torch.load('./D_MATGANIP_energy.pkl'))

