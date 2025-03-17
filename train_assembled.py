import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
import csv
import numpy as np
import pandas as pd
from PIL import Image 
import tensorflow as tf, keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from random import randint
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.metrics import classification_report, confusion_matrix

INPUT_SHAPE_1 = (78,86,1)
INPUT_SHAPE_2 = (256,1)

def get_layer_description(nol: int):
    """
    Returns layer description, based on the number of layers. Each layer is described as size of filter,
    number of filters and the use of pooling layer.
    """
    description = []
    l1 = (nol+2)//3
    l2 = nol-(nol+2)//3
    for i in range(nol):
        if i < l1:
            description.append({"sof":7, "nof":16, "pl": False})
        elif i < l2:
             description.append({"sof":5, "nof":32, "pl": False})
        else:
            description.append({"sof":3, "nof":64, "pl": True})
    return description

def get_optimizer(op:str, lr:float):
    optimizer = None
    if op == "adam":
        optimizer = Adam(learning_rate=lr)
    elif op == "rmsprop":
        optimizer = RMSprop(learning_rate=lr)
    elif op == "sgd":
        optimizer = SGD(learning_rate=lr)
    return optimizer

def create_model_1(params:dict, lr:float):
    """
    Creates a tensorflow/keras model for binary classification using the following parameters:
    :nol: number of layers.
    :af: activation function.
    :op: optimizer.
    :lo: loss function.
    """
    nol = params["nol"]
    nod = params["nod"]
    af = params["af"]
    op = params["op"]
    lo = params["lo"]

    model = keras.Sequential()

    # Adds Conv2D layers
    ld = get_layer_description(nol)
    for i in range(nol):
        nof = ld[i]["nof"]
        sof = ld[i]["sof"]
        pl = ld[i]["pl"]
        if i==0:
            model.add(layers.Conv2D(nof, (sof,sof), activation=af, input_shape=INPUT_SHAPE_1))
        else:
            model.add(layers.Conv2D(nof, (sof,sof), activation=af))
            if pl:
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Adds flattening layer
    model.add(layers.Flatten())
    
    # Adds dense and dropout layers
    for i in range(nod):
        model.add(layers.Dense(256, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))

    # Compiles model
    model.compile(optimizer=get_optimizer(op,lr), 
              loss=lo, 
              metrics=['accuracy'])
    
    return model

def create_model_2(params:dict, lr:float):
    """
    Creates a tensorflow/keras model for binary classification using the following parameters:
    :nol: number of layers.
    :af: activation function.
    :op: optimizer.
    :lo: loss function.
    """
    nol = params["nol"]
    nod = params["nod"]
    af = params["af"]
    op = params["op"]
    lo = params["lo"]

    model = keras.Sequential()

    # Adds Conv1D layers
    ld = get_layer_description(nol)
    for i in range(nol):
        nof = ld[i]["nof"]
        sof = ld[i]["sof"]
        pl = ld[i]["pl"]
        if i==0:
            model.add(layers.Conv1D(nof, sof, activation=af, input_shape=INPUT_SHAPE_2))
        else:
            model.add(layers.Conv1D(nof, sof, activation=af))
    
    # Adds flattening layer
    model.add(layers.Flatten())
    
    # Adds dense and dropout layers
    for i in range(nod):
        model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compiles model
    model.compile(optimizer=get_optimizer(op,lr), 
              loss=lo, 
              metrics=['accuracy'])

    return model

def get_model(lr, model, stage, params):
    compiled_model = None

    if model == 1:
        compiled_model = create_model_1(params,lr)
    elif model == 2:
        compiled_model = create_model_2(params,lr)
    
    return compiled_model

def oversampling(images: list, labels: list, target: int):
    """
    Oversamples dataset.
    """
    cur = len(images)
    while len(images)<target:
        rand = randint(0,cur-1)
        images.append(images[rand])
        labels.append(labels[rand])

def split(images, labels):
    ret = {}
    for i in range(len(images)):
        if labels[i] in ret:
            ret[labels[i]].append(images[i])
        else:
            ret[labels[i]] = [images[i]]
    return ret

def get_dataset(images, labels, model, stage):
    dataset_path = ""
    if model == 1:
        dataset_path = "./dataset_normalized_1"
    elif model == 2:
        dataset_path = "./dataset_normalized_2"

    get_label = {
        1: {
            1:{ 0:0, 1:1, 2:1, 2:1, },
            2:{ 1:1, 2:1, 3:0, },
            3:{ 1:0, 2:1 },
        },
        2:{
            1:{ 0:0, 1:1, 2:1, 2:1, },
            2:{ 1:1, 2:1, 3:0, },
            3:{ 1:0, 2:1 },
        },
        3: {1:0,1:1,2:2,3:3,},
    }

    images_ret = []
    labels_ret = []
    for idx in range(len(images)):
        image = np.array(Image.open(os.path.join(dataset_path,f"{images[idx]}.jpg")))
        get_label_2 = {}
        if model == 1 or model == 2:
            get_label_2  = get_label[model][stage]
        else:
            get_label_2 = get_label["3"]
            
        added = False
        for key, val in get_label_2.items():
            if labels[idx] == key:
                labels_ret.append(val)
                added = True
        if not added:
            continue

        if model == 1:
            images_ret.append(image)
        elif model == 2:
            images_ret.append(np.histogram(image, bins=256, range=(0, 255), density=True)[0])
    
    return images_ret, labels_ret

def get_raw_dataset():
    csv_file = "./dataset/labels.csv"
    images = []
    labels = []

    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            for label in ["0", "1", "2", "3"]:
                if row[1].count(label)>0:
                    images.append(row[0])
                    labels.append(int(label))
                    continue
    
    images, images_test, labels, labels_test = train_test_split(images,labels,test_size=0.2,stratify=labels)
    images_train, images_val, labels_train, labels_val = train_test_split(images,labels,test_size=0.2,stratify=labels)
    return images_train, images_val, images_test, labels_train, labels_val, labels_test

def get_normalized(images, model):
    dataset_path = ""
    if model == 1:
        dataset_path = "./dataset_normalized_1"
    elif model == 2:
        dataset_path = "./dataset_normalized_2"

    images_ret = []
    for idx in range(len(images)):
        image = np.array(Image.open(os.path.join(dataset_path,f"{images[idx]}.jpg")))

        if model == 1:
            images_ret.append(image)
        elif model == 2:
            images_ret.append(np.histogram(image, bins=256, range=(0, 255), density=True)[0])
    
    return images_ret

def test_assembled(models, stages, images_test, labels_test):
    paths = ["./results/assembled/aux-1.keras", "./results/assembled/aux-2.keras", "./results/assembled/aux-3.keras"]
    loaded_models = [tf.keras.models.load_model(path) for path in paths]
    
    idx_to_label = [0,3,1]
    predicted_labels = []
    for idx_image in range(len(images_test)):
        predicted = False
        for idx_model in range(len(loaded_models)):
            image = get_normalized([images_test[idx_image]],models[idx_model])
            if(len(image)==0):
                predicted=True
                break
            prediction = loaded_models[idx_model].predict(np.array(image), verbose=0)[0][0]
            if prediction < 0.5:
                predicted_labels.append(idx_to_label[idx_model])
                predicted = True
                break
        if (not predicted): 
            predicted_labels.append(2)
    
    return labels_test, predicted_labels

def train(lrs, epochs, models, stages, params):
    images_train, images_val, images_test, labels_train, labels_val, labels_test = get_raw_dataset()

    for idx in range(3):
        images_train_2, labels_train_2 = get_dataset(images_train, labels_train, models[idx], stages[idx])

        maximum = 0
        images_aux = []
        labels_aux = []
        for key, value in split(images_train_2, labels_train_2).items():
            maximum = max(maximum, len(value))
        for key, value in split(images_train_2, labels_train_2).items():
            images_to_add = value
            labels_to_add = [key]*len(value)
            oversampling(images_to_add,labels_to_add,maximum)
            images_aux.extend(images_to_add)
            labels_aux.extend(labels_to_add)
        images_train_2 = images_aux
        labels_train_2 = labels_aux

        compiled_model = get_model(lrs[idx],models[idx],stages[idx],params[idx])

        checkpoint_callback = ModelCheckpoint(
            f"./results/assembled/aux-{stages[idx]}.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        )

        images_val_2, labels_val_2 = get_dataset(images_val, labels_val, models[idx], stages[idx])
        history = compiled_model.fit(
            np.array(images_train_2),
            np.array(labels_train_2),
            shuffle=True,
            epochs=epochs[idx],
            validation_data=(np.array(images_val_2), np.array(labels_val_2)),
            callbacks = [checkpoint_callback]
        )
    
    return test_assembled(models, stages, images_test, labels_test)

def show_graph(labels, predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    confu_matrix = confusion_matrix(labels,predictions)

    sea.heatmap(confu_matrix, annot=True, fmt="d", cmap="plasma", ax=axes[0])
    axes[0].set_title("Matriz de confusion")
    axes[0].set_xlabel("Etiquetas predichas")
    axes[0].set_ylabel("Etiquetas reales")

    class_report = classification_report(labels,predictions,output_dict=True)
    print(class_report)
    report_df = pd.DataFrame(class_report).transpose()
    class_metrics = report_df.drop(["accuracy", "macro avg", "weighted avg"])
    class_metrics.rename(columns={"precision": "Precision","recall": "Sensibilidad","f1-score": "Puntaje F1"}, inplace=True)
    bar_chart = class_metrics[["Precision", "Sensibilidad", "Puntaje F1"]].plot(kind="bar", ax=axes[1], colormap="plasma", rot=0)
    axes[1].set_title("Reporte de clasificacion por etiqueta")
    axes[1].set_xlabel("Etiquetas")
    axes[1].set_ylabel("Puntaje")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y")

    for container in bar_chart.containers:
        for bar in container:
            height = bar.get_height()
            bar_chart.annotate(
                f"{height:.2f}", 
                xy=(bar.get_x() + bar.get_width() / 2, height), 
                xytext=(0, 3),
                textcoords="offset points", 
                ha="center", va="bottom",
                rotation=90
            )

    path = "./results/assembled/aux.png"
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
    plt.savefig(path)
    plt.show()

def clean():
    path = f"./results/assembled/assembled.png"
    if os.path.exists(path):
        os.remove(path)

    for idx in range(3):
        path = f"./results/assembled/{idx+1}.keras"
        if os.path.exists(path):
            os.remove(path)

def save_model():
    shutil.copy(
        "./results/assembled/aux.png",
        "./results/assembled/assembled.png"
    )

    for idx in range(3):
        shutil.copy(
            f"./results/assembled/aux-{idx+1}.keras",
            f"./results/assembled/{idx+1}.keras"
        )

def get_best(params, models, stages, lrs, epochs):
    finished = False

    while not finished:
        labels, predictions, = train(lrs,epochs,models,stages,params)

        show_graph(labels,predictions)

        save = str(input("Save model (Y/N)? "))
        if save.upper() == "Y":
            clean()
            save_model()

        finished = str(input("Continue (Y/N)? ")).upper()=="N"

params = [
    {"nol":7, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"},
    {"nol":3, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"},
    {"nol":3, "nod":2, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
]
models = [2, 2, 2]
stages = [1, 2, 3]
lrs = [0.0001,0.000065,0.0006]
epochs = [20,30,60]

# {
#     '0': {'precision': 0.9753086419753086, 'recall': 0.9634146341463414, 'f1-score': 0.9693251533742331, 'support': 82.0}, 
#     '1': {'precision': 0.6, 'recall': 0.75, 'f1-score': 0.6666666666666666, 'support': 24.0}, 
#     '2': {'precision': 0.5, 'recall': 0.2857142857142857, 'f1-score': 0.36363636363636365, 'support': 7.0}, 
#     '3': {'precision': 0.8148148148148148, 'recall': 0.7586206896551724, 'f1-score': 0.7857142857142857, 'support': 29.0}, 
#     'accuracy': 0.852112676056338, 
#     'macro avg': {'precision': 0.7225308641975309, 'recall': 0.6894374023789498, 'f1-score': 0.6963356173478873, 'support': 142.0}, 
#     'weighted avg': {'precision': 0.8556685793774995, 'recall': 0.852112676056338, 'f1-score': 0.8508157141398306, 'support': 142.0}
# }
get_best(params, models, stages, lrs, epochs)
