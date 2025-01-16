import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import ast
import csv
from PIL import Image 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf, keras
from tensorflow.keras.models import load_model
import seaborn as sea
from sklearn.metrics import classification_report, confusion_matrix


def create_df(params: dict):
    columns = ["model", "val_acc", "tr_acc"]
    for key in params.keys():
        columns.append(key)
    return pd.DataFrame(columns=columns)

def create_graph(path: str, name: str):
    df = pd.DataFrame()

    with open(os.path.join(path,name), mode='r') as file:
        csv_reader = csv.reader(file)
        
        keys = []
        for row in csv_reader:
            params = ast.literal_eval(row[0])
            keys = list(params.keys()) 
            training_accuracy = ast.literal_eval(row[1])
            val_accuracy = ast.literal_eval(row[2])
            
            idx = val_accuracy.index(max(val_accuracy))
            val_acc = val_accuracy[idx]
            tr_acc = training_accuracy[idx]

            if df.empty:
                df = create_df(params)
            
            params["val_acc"] = val_acc
            params["tr_acc"] = tr_acc
            df = pd.concat([df, pd.Series(params).to_frame().T],ignore_index=True)
    
    df = df.groupby(keys,as_index=False).agg({"val_acc":"mean","tr_acc":"mean"})

    fig, axs = plt.subplots((len(keys)+2)//3,3,figsize=(20,10))
    for idx, column in enumerate(keys):
        labels = np.unique(df[column])
        colors = plt.cm.viridis(np.linspace(0,1,len(labels)))
        map_color = {label: colors[i] for i,label in enumerate(labels)}

        for label in labels:
            axs[idx//3, idx%3].scatter(df[df[column]==label]["tr_acc"],df[df[column]==label]["val_acc"],color=map_color[label],label=label)
            axs[idx//3, idx%3].set_title(f"Accuracy by {column}")
            axs[idx//3, idx%3].set_xlabel("Training accuracy")
            axs[idx//3, idx%3].set_ylabel("Validation accuracy")
            axs[idx//3, idx%3].set_xlim([0,1])
            axs[idx//3, idx%3].set_ylim([0,1])
            axs[idx//3, idx%3].legend()
    plt.tight_layout()
    plt.legend()
    print(os.path.join(path,name.strip(".")[0]+".png"))
    plt.savefig(os.path.join(path,name.split(".")[0]+".png"))
    plt.clf()

def create_graphs(path: str):
    for filename in os.listdir(path):
        create_graph(path, filename)

create_graphs("./results/complete_search")

def get_candidates(path: str):
    df = pd.DataFrame()

    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        
        keys = []
        for row in csv_reader:
            params = ast.literal_eval(row[0])
            keys = list(params.keys()) 
            training_accuracy = ast.literal_eval(row[1])
            val_accuracy = ast.literal_eval(row[2])
            
            idx = val_accuracy.index(max(val_accuracy))
            val_acc = val_accuracy[idx]
            tr_acc = training_accuracy[idx]

            if df.empty:
                df = create_df(params)
            
            params["val_acc"] = val_acc
            params["tr_acc"] = tr_acc
            df = pd.concat([df, pd.Series(params).to_frame().T],ignore_index=True)
    
    df = df.groupby(keys,as_index=False).agg({"val_acc":"mean","tr_acc":"mean"})
    df = df.sort_values(["val_acc","tr_acc"],ascending=[False,False])
    df = df.iloc[:10]
    print(df)
    print("")

get_candidates("./results/complete_search/1-1.csv")
get_candidates("./results/complete_search/1-2.csv")
get_candidates("./results/complete_search/1-3.csv")
get_candidates("./results/complete_search/2-1.csv")
get_candidates("./results/complete_search/2-2.csv")
get_candidates("./results/complete_search/2-3.csv")
get_candidates("./results/complete_search/3.csv")
get_candidates("./results/complete_search/4.csv")
