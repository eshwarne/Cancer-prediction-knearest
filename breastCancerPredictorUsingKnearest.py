import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd;
df=pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.head())