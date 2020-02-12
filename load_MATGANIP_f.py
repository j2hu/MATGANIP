
# coding: utf-8

# In[ ]:packages in MATGANIP.f


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


global sample_num, rmat_num
sample_num=1
rmat_num=28
patt_xrd = xrd.XRDCalculator('CuKa')


# In[ ]:to define functions



def get_tfactor2(mgStructure):
    ppp=mgStructure.species
    ttt4=set()
    xxx=[]
    for i in ppp:
        if i not in ttt4:
            xxx.append(i)
        ttt4.add(i)
    if len(xxx)==3:
        E_A_r=xxx[0].ionic_radius
        E_B_r=xxx[1].ionic_radius
        E_X_r=xxx[2].ionic_radius
    else:
        print(error)
        
    sum1 = E_A_r+E_X_r
    sum2 = np.sqrt(2)*(E_B_r+E_X_r)
    Tfactor=sum1/sum2
    Tfactor=Tfactor.real
    return Tfactor 

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

def cif_mat3_couple(Random_Structure):
    global rmat_num
    RS_xrdmat = get_xrdmat3(Random_Structure)
 
    multimat3_RS =  np.zeros((rmat_num,rmat_num),dtype='float32')
    multimat3_RS = np.asarray(np.dot(RS_xrdmat.T, RS_xrdmat))
    return multimat3_RS

def deal_randomcif(input_path):

    folder=input_path
    depth_filename=np.random.choice(glob.glob(folder+"*"))
    Random_mgStructure = mg.Structure.from_file(depth_filename)
    return Random_mgStructure

def get_tfactor(myStructure):
    total_point=myStructure.species
    dereplication=set()
    final3=[]
    for i in total_point:
        if i not in dereplication:
            final3.append(i)
        dereplication.add(i)
    if len(final3)==3:
        E_A_r=final3[0].ionic_radius
        E_B_r=final3[1].ionic_radius
        E_X_r=final3[2].ionic_radius
    else:
        print(error)
        
    sum1 = E_A_r+E_X_r
    sum2 = np.sqrt(2)*(E_B_r+E_X_r)
    Tfactor=sum1/sum2
    Tfactor=Tfactor.real
    return Tfactor 

def cal_xrdmat(mgStructure):
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

def GANs_Gmat(Random_Structure):
    global rmat_num
    RS_xrdmat = cal_xrdmat(Random_Structure)
    multimat3_RS =  np.zeros((rmat_num,rmat_num),dtype='float32')
    multimat3_RS = np.asarray(np.sqrt(np.dot(RS_xrdmat.T, RS_xrdmat)))
    return multimat3_RS


# In[ ]: to define Generator and Discriminator 


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

        
    def forward(self,x):
        D_out,(h_n,h_c)=self.Dlstm(x,None)
        out = self.out(D_out[:,-1,:]) #(batch,time step,input)   
        return out


# In[ ]: to build G and D; to load trained MATGANIP.f


G=GNet()
D=DNet()
G.load_state_dict(torch.load('./G_MATGATIP_factor.pkl'))
D.load_state_dict(torch.load('./D_MATGANIP_factor.pkl'))

