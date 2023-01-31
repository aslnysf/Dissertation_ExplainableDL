#%%
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from numpy import array
import pickle
import pandas as pd
from pandas import read_csv

layer = 3
epoch = 200

file_name = './I_XYT_drop_maxpooling.csv'

ixyt = read_csv(file_name)

ixt = []
iyt = []

ixt.append((ixyt.loc [:,'IXT D0']).values)
ixt.append((ixyt.loc [:,'IXT D10']).values)
ixt.append((ixyt.loc [:,'IXT D20']).values)
ixt.append((ixyt.loc[:, 'IXT D30']).values)
ixt.append((ixyt.loc[:, 'IXT D50']).values)

iyt.append((ixyt.loc [:,'IYT D0']).values)
iyt.append((ixyt.loc [:,'IYT D10']).values)
iyt.append((ixyt.loc [:,'IYT D20']).values)
iyt.append((ixyt.loc[:, 'IYT D30']).values)
iyt.append((ixyt.loc[:, 'IYT D50']).values)

fig = plt.figure(dpi=1200)
plt.xlabel('Epochs')
plt.ylabel('Mutual Information I(X;T) with Dropout')
#plt.plot(ixt[0], label=' 0% Dropout')
plt.plot(ixt[1], label='10% Dropout')
plt.plot(ixt[2], label='20% Dropout')
plt.plot(ixt[3], label='30% Dropout')
plt.plot(ixt[4], label='50% Dropout')
plt.legend()
plt.tight_layout();
fig.savefig("./mutuInfo_IXT_drop_maxpooling_2.png", dpi=1200)
plt.show()
    
fig = plt.figure(dpi=1200)
plt.xlabel('Epochs')
plt.ylabel('Mutual Information I(Y;T) with Dropout')
#plt.plot(iyt[0], label='0% Dropout')
plt.plot(iyt[1], label='10% Dropout')
plt.plot(iyt[2], label='20% Dropout')
plt.plot(iyt[3], label='30% Dropout')
plt.plot(iyt[4], label='50% Dropout')
plt.legend()
plt.tight_layout();
fig.savefig("./mutuInfo_IYT_drop_maxpooling_2.png", dpi=1200)
plt.show()

# ixyt1 = read_csv('ixyt1.csv')
# ixyt2 = read_csv('ixyt2.csv')

# ixyt3 = read_csv('ixyt3.csv')
# ixyt4 = read_csv('ixyt4.csv')

# ixytL1 = (ixyt1 + ixyt2)/2
# ixytL1 = pd.DataFrame(ixytL1)
# ixytL2 = (ixyt3 + ixyt4)/2
# ixytL2 = pd.DataFrame(ixytL2)

# ixtL1 = ixytL1.loc[:,'I_XT']
# iytL1 = ixytL1.loc[:, 'I_YT']
# ixtL2 = ixytL2.loc[:,'I_XT']
# iytL2 = ixytL2.loc[:, 'I_YT']

# #ixyt.to_csv('ixyt.csv')
# epoch = 100
# ixt_array = []
# iyt_array = []
# ixt_array.append(ixtL1.values)
# ixt_array.append(ixtL2.values); 
# iyt_array.append(iytL1.values)
# fig = plt.figure(dpi=600)
# plt.xlabel('Epochs')
# plt.ylabel('Mutual Information')
# plt.plot(ixt_array[0], color='blue', label='Layer1')
# plt.plot(ixt_array[1], color='red', label='Layer2')
# #plt.plot(iyt_array[0], color='red',  label='IYT')
# plt.legend()
# plt.tight_layout();
# #fig.savefig(path_figures + "accuracy.png", dpi=600)
# plt.show()