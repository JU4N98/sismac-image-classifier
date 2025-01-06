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

        # cmap = plt.get_cmap("Set1")  
        # labels = np.unique(df[column])
        # colors = [cmap(i / len(labels)) for i in range(len(labels))]
        # map_color = {label: colors[i] for i, label in enumerate(labels)}

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

# create_graphs("./models/model_1")
# create_graphs("./models/model_2")
# create_graphs("./models/model_3")
# create_graphs("./models/model_4")

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

# get_candidates("./models/model_1/results_000.csv")
# get_candidates("./models/model_1/results_111.csv")
# get_candidates("./models/model_1/results_222.csv")
# get_candidates("./models/model_2/results_000.csv")
# get_candidates("./models/model_2/results_111.csv")
# get_candidates("./models/model_2/results_222.csv")
# get_candidates("./models/model_3/results_000.csv")
# get_candidates("./models/model_4/results_000.csv")

def load_binary_model(paths):
    return [tf.keras.models.load_model(path) for path in paths]

def load_multiclass_model(path):
    return tf.keras.models.load_model(path)

def load_model(model):
    if model == 1:
        paths = [
            "./models/manual_training/1-1-3.keras",
            "./models/manual_training/1-2-1.keras", 
            "./models/manual_training/1-3-2.keras"
        ]
        return load_binary_model(paths)
    elif model == 2:
        paths = [
            "./models/manual_training/2-1-2.keras", 
            "./models/manual_training/2-2-3.keras", 
            "./models/manual_training/2-3-2.keras"
        ]
        return load_binary_model(paths)
    elif model == 3:
        path = "./models/manual_training/3-0-3.keras"
        return load_multiclass_model(path)
    elif model == 4:
        path = "./models/manual_training/4-0-3.keras"
        return load_multiclass_model(path)

def get_dataset(model):
    csv_file = "./dataset/labels.csv"
    dataset_path = ""
    images = []
    labels = []
    if model == 1:
        dataset_path = "./dataset_normalized_1"
    elif model == 2:
        dataset_path = "./dataset_normalized_2"
    elif model == 3:
        dataset_path = "./dataset_normalized_1"
    elif model == 4:
        dataset_path = "./dataset_normalized_1"

    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            label = str(row[1])
            if label.count("0"):
                labels.append(0)
            elif label.count("1"):
                labels.append(1)
            elif label.count("2"):
                labels.append(2)
            elif label.count("3"):
                labels.append(3)
            else:
                continue

            image = np.array(Image.open(os.path.join(dataset_path,row[0]+".jpg")))
            if model == 1:
                images.append(image)
            elif model == 2:
                images.append(np.histogram(image, bins=256, range=(0, 255), density=True)[0])
            elif model == 3:
                image = tf.image.grayscale_to_rgb(tf.convert_to_tensor(np.expand_dims(image, axis=-1)))
                image = tf.image.resize_with_pad(image, target_height = 224, target_width = 224)
                images.append(image)
            elif model == 4:
                image = tf.image.grayscale_to_rgb(tf.convert_to_tensor(np.expand_dims(image, axis=-1)))
                image = tf.image.resize_with_pad(image, target_height = 299, target_width = 299)
                images.append(image)
    return images, labels

def get_binary_prediction(models, image):
    idx_to_label = [0,3,1]
    for idx, model in enumerate(models):
        prediction = model.predict(image, verbose=0)[0][0]
        if prediction < 0.5:
            return idx_to_label[idx]
    return 2

def get_multiclass_prediction(model, image):
    return np.argmax(model.predict(image, verbose=0),axis=-1)[0]

def test_models():
    for m in range(1,5):
        model = load_model(m)
        images, labels = get_dataset(m)
        predictions = []
        for image in images:
            if m <= 2:
                predictions.append(get_binary_prediction(model,np.array([image])))
            else:
                predictions.append(get_multiclass_prediction(model,np.array([image])))
        class_report = classification_report(labels,predictions)
        confu_matrix = confusion_matrix(labels,predictions)
        print(class_report)
        print(confu_matrix)

test_models()