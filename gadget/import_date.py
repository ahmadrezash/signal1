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
# p = Path().joinpath('data', 'New_Shuffled_Train(disorder)', 'subj_1.mat')

# %%
def load_data():
	mats = []
	x1 = []
	y1 = []
	x2 = []
	y2 = []

	disorder = Path('./').parent.absolute().joinpath('data', 'New_Shuffled_Train(disorder)')
	normal = Path('./').parent.absolute().joinpath('data', 'New_Shuffled_Train_normal')
	for file in os.listdir(normal):
		x1.append(loadmat(str(normal) + '/' + file)[file[:-4]].T)
		y1.append(np.ones((295, 19, 2)))
	for file in os.listdir(disorder):
		x2.append(loadmat(str(disorder) + '/' + file)[file[:-4]].T)
		y2.append(np.zeros((295, 19, 2)))
	# print(a.shape[0])
	return x1, y1, x2, y2
