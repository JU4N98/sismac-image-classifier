import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/labels.csv", sep=",")
tgt = df["label"]

sorted_tgt = tgt.sort_values()

sea.countplot(x=sorted_tgt, order=sorted_tgt.unique()) 
plt.title("Histograma de frecuencias de clases")
plt.xlabel("Clases")
plt.ylabel("Numero de imagenes")
plt.show()
