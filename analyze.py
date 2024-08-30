import numpy as np
import pandas as pd 
import seaborn as sea 
import matplotlib.pyplot as plt 

df = pd.read_csv("dataset/labels.csv", sep=",")
tgt = df["label"]
sea.countplot(x=tgt)
plt.show()
