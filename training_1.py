import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf, keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from random import randint
import matplotlib.pyplot as plt

""" 
Parameters of the model: 
* number of layers: [2,10]
* size of the kernel/filter of the ith layer: 7x7 (primeras capas), 5x5 (capas medias), 3x3 (ultimas capas)
* number of kernels of the ith layer: 16, 32, 64 (primeras capas), 64, 128, 256 (capas medias) y 256, 512, 1024 (ultimas capas)
* activation function: (ReLu, Sigmoid, Tanh)
* usage of pooling layers and kind of them: si o no, que tipo
* optimizer: SGD, Adam, RMSProp
* loss: categorical cross entropy (capaz en la segunda NN), binary cross entropy (en la primera y capaz en la segunda NN)
* metrics: accuracy

Models to try:
* model #1: use CNN using the images as input and:
    * first classify images among "sin defectos" and "con defectos"
    * finally classify images belonging to "con defectos" among "sobrecarga en 1 fase", 
    "sobrecarga en 2 fase" and "sobrecarga en 3 fase"
* model #2: use MLP or CNNs using a histogram of the image as input and try to do 
something similar to the previous case.

As an alternative it's also possible to make the whole classification process in the same
neural network.
"""

INPUT_SHAPE_1 = (78,86,1)
INPUT_SHAPE_2 = (256,1)
K_FOLD = 5
EPOCHS = 15
# RANDOM_STATE = 13
kfold = StratifiedKFold(n_splits=K_FOLD, shuffle=True)

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

def create_model_1(params:dict):
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
    # model.add(layers.Dropout(0.5))
    for i in range(nod):
        model.add(layers.Dense(256, activation="relu"))
        # model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation="sigmoid"))

    # Compiles model
    model.compile(optimizer=op, 
              loss=lo, 
              metrics=['accuracy'])
    
    model.summary()

    return model

def normalize(images):
    normalized = []
    for image in images:
        normalized.append(image/256)
    return normalized

def train_model_1(model, images_train, labels_train, images_val, labels_val, f):
    checkpoint_callback = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    )
    return model.fit(
        np.array(normalize(images_train[f])),
        np.array(labels_train[f]),
        shuffle=True,
        epochs=EPOCHS,
        validation_data=(np.array(normalize(images_val[f])), np.array(labels_val[f])),
        callbacks = [checkpoint_callback]
    )

def test_model_1(images, labels):
    best_model = models.load_model('best_model.keras')
    predictions = best_model.predict(np.array(normalize(images)))
    predictions =  (predictions >= 0.5).astype(int)
    # class_report = classification_report(labels,predictions)
    # confu_matrix = confusion_matrix(labels,predictions)
    test_loss, test_accuracy = best_model.evaluate(np.array(normalize(images)), np.array(labels))
    return [test_accuracy, test_loss] # class_report, confu_matrix]

def create_model_2(params:dict):
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
            # if pl:
            #     model.add(layers.MaxPooling1D(pool_size=2))
    
    # Adds flattening layer
    model.add(layers.Flatten())
    
    # Adds dense and dropout layers
    # model.add(layers.Dropout(0.5))
    for i in range(nod):
        model.add(layers.Dense(256, activation="relu"))
        # model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation="sigmoid"))

    # Compiles model
    model.compile(optimizer=op, 
              loss=lo, 
              metrics=['accuracy'])
    
    model.summary()

    return model

def get_histograms(images):
    histograms = []
    for image in images:
        histograms.append(np.histogram(image, bins=256, range=(0, 255), density=True)[0])
    return np.array(histograms)

def train_model_2(model, images_train, labels_train, images_val, labels_val, f):
    checkpoint_callback = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    )
    return model.fit(
        get_histograms(images_train[f]),
        np.array(labels_train[f]),
        shuffle=True,
        epochs=EPOCHS,
        validation_data=(get_histograms(images_val[f]), np.array(labels_val[f])),
        callbacks = [checkpoint_callback]
    )

def test_model_2(images, labels):
    best_model = models.load_model('best_model.keras')
    predictions = best_model.predict(get_histograms(images))
    predictions =  (predictions >= 0.5).astype(int)
    # class_report = classification_report(labels,predictions)
    # confu_matrix = confusion_matrix(labels,predictions)
    test_loss, test_accuracy = best_model.evaluate(get_histograms(images), np.array(labels))
    return [test_accuracy, test_loss] # class_report, confu_matrix]

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

def save_history(history, name: str):
    """
    Saves a plot with the accuracy and validation accuracy of a model through the epochs.
    """
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(name+".png")
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.clf()

def save_row(path:str, new_row:list):
    """
    Appends a row in a csv with the model description, accuracy and validation accuracy.
    """
    print(new_row)
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_row)

def train(params: dict, images: list, labels: list, create_model,train_model,test_model,path:str):
    """
    Trains and evaluates the model through each fold.
    """
    # Creates folds and balances labels
    images_test = []
    labels_test = []
    images, images_test, labels, labels_test = train_test_split(images,labels,test_size=0.2,stratify=labels)

    images_train = [[] for _ in range(K_FOLD)]
    labels_train = [[] for _ in range(K_FOLD)]
    images_val = [[] for _ in range(K_FOLD)]
    labels_val = [[] for _ in range(K_FOLD)]
    fold = 0
    for train_idx, val_idx in kfold.split(images, labels):
        maximum = 0
        for key, value in split(np.array(images)[train_idx], np.array(labels)[train_idx]).items():
            maximum = max(maximum, len(value))
        for key, value in split(np.array(images)[train_idx], np.array(labels)[train_idx]).items():
            images_to_add = value
            labels_to_add = [key]*len(value)
            oversampling(images_to_add,labels_to_add,maximum)
            images_train[fold].extend(images_to_add)
            labels_train[fold].extend(labels_to_add)
        images_val[fold].extend(np.array(images)[val_idx])
        labels_val[fold].extend(np.array(labels)[val_idx])
        fold += 1

    # trains and evaluates the model
    for f in range(K_FOLD):
        model = create_model(params)
        history = train_model(model,images_train,labels_train,images_val,labels_val,f)
        row = [str(params),history.history["accuracy"],history.history["val_accuracy"],history.history["loss"],history.history["val_loss"]]
        row.extend(test_model(images_test, labels_test))
        save_row(path,row)

def get_dataset_0(path:str):
    """
    Returns dataset for the #1 CNNs which classifies images between "sin defectos" (0) and "con defectos" (1).
    """
    csv_file = "./dataset/labels.csv"
    images = []
    labels = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            image = np.array(Image.open(os.path.join(path,row[0]+".jpg")))
            if row[1].count("0"):
                labels.append(0)
                images.append(image)
            elif row[1].count("1") or row[1].count("2") or row[1].count("3"):
                labels.append(1)
                images.append(image)
    return images, labels

def get_dataset_1(path:str):
    """
    Returns dataset for the #2 CNNs which classifies images between "sobrecarga en 3 fases" (0) and "sobrecarga en mas de 1 o 2 fases" (1).
    """
    csv_file = "./dataset/labels.csv"
    images = []
    labels = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            image = np.array(Image.open(os.path.join(path,row[0]+".jpg")))
            if row[1].count("3"):
                labels.append(0)
                images.append(image)
            elif row[1].count("1") or row[1].count("2"):
                labels.append(1)
                images.append(image)
    return images, labels

def get_dataset_2(path:str):
    """
    Returns dataset for the #3 CNNs which classifies images between "sobrecarga en 1 fase" (0) and "sobrecarga en 2 fases" (1).
    """
    csv_file = "./dataset/labels.csv"
    images = []
    labels = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            image = np.array(Image.open(os.path.join(path,row[0]+".jpg")))
            if row[1].count("1"):
                labels.append(0)
                images.append(image)
            elif row[1].count("2"):
                labels.append(1)
                images.append(image)
    return images, labels

def backtracking(idx: int, parameters: list, values: dict, chosen: dict, images: list, labels: list, create_model, train_model, test_model, path:str):
    """
    Evaluates each of the models that can be get from all combinations of parameters values.
    """
    if idx == len(parameters):
        train(chosen,images,labels,create_model,train_model,test_model,path)
        return
    
    for val in values[parameters[idx]]:
        chosen[parameters[idx]] = val
        backtracking(idx+1,parameters,values,chosen,images,labels,create_model,train_model,test_model,path)
    
    return

# Model 1: convolutional 2D neural netwrok with binary classification

parameters = ["nol", "nod", "af", "op", "lo"]
parameters_values = {
    "nol" : [3,4,5,6,7,8,9,10], # number of layers
    "nod" : [2,3,4], # number of dense layers
    "af" : ["relu"], # activation function
    "op" : ["sgd","adam","rmsprop"], # optimizer
    "lo" : ["binary_crossentropy"] # loss function
}

# chosen = {}
# images, labels = get_dataset_0("./dataset_normalized_1")
# backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_1,train_model_1,test_model_1,"./models/model_1/results_00.csv")

# chosen = {}
# images, labels = get_dataset_1("./dataset_normalized_1")
# backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_1,train_model_1,test_model_1,"./models/model_1/results_11.csv")

# chosen = {}
# images, labels = get_dataset_2("./dataset_normalized_1")
# backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_1,train_model_1,test_model_1,"./models/model_1/results_22.csv")

chosen = {}
images, labels = get_dataset_0("./dataset_normalized_1")
backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_1,train_model_1,test_model_1,"./models/model_1/results_000.csv")

chosen = {}
images, labels = get_dataset_1("./dataset_normalized_1")
backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_1,train_model_1,test_model_1,"./models/model_1/results_111.csv")

chosen = {}
images, labels = get_dataset_2("./dataset_normalized_1")
backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_1,train_model_1,test_model_1,"./models/model_1/results_222.csv")

# Model 2: convolutional 1D neural netwrok with binary classification

# chosen = {}
# images, labels = get_dataset_0("./dataset_normalized_2")
# backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_2,train_model_2,test_model_2,"./models/model_2/results_00.csv")

# chosen = {}
# images, labels = get_dataset_1("./dataset_normalized_2")
# backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_2,train_model_2,test_model_2,"./models/model_2/results_11.csv")

# chosen = {}
# images, labels = get_dataset_2("./dataset_normalized_2")
# backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_2,train_model_2,test_model_2,"./models/model_2/results_22.csv")

chosen = {}
images, labels = get_dataset_0("./dataset_normalized_2")
backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_2,train_model_2,test_model_2,"./models/model_2/results_000.csv")

chosen = {}
images, labels = get_dataset_1("./dataset_normalized_2")
backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_2,train_model_2,test_model_2,"./models/model_2/results_111.csv")

chosen = {}
images, labels = get_dataset_2("./dataset_normalized_2")
backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_2,train_model_2,test_model_2,"./models/model_2/results_222.csv")
