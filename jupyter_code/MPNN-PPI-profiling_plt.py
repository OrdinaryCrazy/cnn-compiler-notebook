#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch_geometric as pyg
import os.path as osp
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI


# In[9]:


CoraProfile = pd.read_excel("../gcn_Cora_profile.xlsx")
CiteSeerProfile = pd.read_excel("../gcn_CiteSeer_profile.xlsx")
PubMedProfile = pd.read_excel("../gcn_PubMed_profile.xlsx")


# In[25]:


plt.figure(figsize=(15,10)),
plt.plot(CoraProfile.epoch, CoraProfile.agg_time, color = "red", linestyle = "--", 
         label = "Cora_agg_time n_v_aver = 7.8, d = 1443")
plt.plot(CoraProfile.epoch, CoraProfile.map_time, color = "red", linestyle = "-", 
         label = "Cora_map_time n_v_aver = 7.8, d = 1443")
plt.plot(CiteSeerProfile.epoch, CiteSeerProfile.agg_time, color = "green", linestyle = "--", 
         label = "CiteSeer_agg_time n_v_aver = 5.5, d = 3703")
plt.plot(CiteSeerProfile.epoch, CiteSeerProfile.map_time, color = "green", linestyle = "-", 
         label = "CiteSeer_map_time n_v_aver = 5.5, d = 3703")
plt.plot(PubMedProfile.epoch, PubMedProfile.agg_time, color = "blue", linestyle = "--", 
         label = "PubMed_agg_time n_v_aver = 9, d = 500")
plt.plot(PubMedProfile.epoch, PubMedProfile.map_time, color = "blue", linestyle = "-", 
         label = "PubMed_map_time n_v_aver = 9, d = 500")
plt.ylabel("second")
plt.xlabel("epoch")
plt.legend(loc='upper right')
plt.savefig("../plot.png")

