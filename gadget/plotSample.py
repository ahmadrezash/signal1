# %%
from pathlib import Path
import seaborn as sns
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os


# %%
p = Path().joinpath('data', 'New_Shuffled_Train(disorder)', 'subj_1.mat')
# mat = loadmat(p)
# %%
# a = mat['subj_1'].T
# %%
# plt.figure(figsize=(200, 80))
# x = plt.plot(a[0], linewidth=5)
# y = plt.plot(a[1], linewidth=5)
# plt.show()

#
# %%
# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10,2))
# plt.plot(a[0], lw=1, c='g')
#
# plt.show()
#
# plt.figure(figsize=(10,2))
# plt.plot(a[1], lw=1, c='b')
# plt.show()


#%%

mats = []
x1 = []
y1 = []
x2 = []
y2 = []

disorder =  Path('./').absolute().joinpath('data','New_Shuffled_Train(disorder)')
normal = Path('./').absolute().joinpath('data','New_Shuffled_Train_normal')
for file in os.listdir( normal ) :
    x1.append( loadmat( str(normal)+'/'+file )[file[:-4]].T )
    y1.append(np.ones((295,19,2)))

for file in os.listdir( disorder ) :
    x2.append( loadmat( str(disorder)+'/'+file )[file[:-4]].T )
    y2.append(np.zeros((295,19,2)))
# print(a.shape[0])
sample = 0
#%%
sample1 = np.random.choice(len(x1), 10)
sample2 = np.random.choice(len(x2), 10)

#%%
import matplotlib.pyplot as plt

for i in sample1:
	fig = plt.figure(figsize=(140,90))
	print(i)
	for j in range(19):
		plt.subplot(19, 1, j+1)
		plt.plot(x1[i][j])
	plt.savefig('./plot' + str(i) + '-.png')
	plt.show()
#%%
for i in sample2:
	fig = plt.figure(figsize=(140,90))
	print(i)
	for j in range(19):
		plt.subplot(19, 1, j+1)
		plt.plot(x2[i][j])
	plt.savefig('./plot' + str(i) + '-.png')
	plt.show()
