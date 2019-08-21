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

#%%
from gadget import import_date
x0, y0, x1, y1 = import_date.load_data()
#%%
lbl = np.array(y0)
