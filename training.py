import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf, keras
from tensorflow.keras import layers, models
import csv
from PIL import Image
import numpy as np

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

INPUT_SHAPE = (234,259,1)

def get_layer_description(nol: int):
    description = []
    l1 = (nol+2)//3
    l2 = nol-(nol+2)//3
    for i in range(nol):
        if i < l1:
            description.append({"sof":7, "nof":16})
        elif i < l2:
             description.append({"sof":5, "nof":32})
        else:
            description.append({"sof":3, "nof":64})
    return description



def create_model(nol: int, nod: int, af: str, op: str, lo: str):
    """
    Creates a tensorflow/keras model using the following parameters:
    :nol: number of layers.
    :af: activation function.
    :op: optimizer.
    :lo: loss function.
    """

    model = keras.Sequential()

    # Adds Conv2D layers
    ld = get_layer_description(nol)
    print(ld)
    for i in range(nol):
        nof = ld[i]["nof"]
        sof = ld[i]["sof"]
        if i==0:
            model.add(layers.Conv2D(nof, (sof,sof), activation=af, input_shape=INPUT_SHAPE))
        else:
            model.add(layers.Conv2D(nof, (sof,sof), activation=af))
    
    # Adds flattening layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    
    # Adds dense and dropout layers
    model.add(layers.Dropout(0.5))
    for i in range(nod):
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(2, activation="softmax"))

    # Compiles model
    model.compile(optimizer=op, 
              loss=lo, 
              metrics=['accuracy'])

    return model

def train(params: dict, images: list, labels: list):
    to_train = create_model(params["nol"],params["nod"],params["af"],params["op"],params["lo"])

def get_dataset_0():
    """
    Returns dataset for the first CNNs which classifies images between "sin defectos" (0) and "con defectos" (1).
    """
    csv_file = "./dataset/labels.csv"
    images = []
    labels = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        image = np.array(Image.open(os.path.join("./dataset_normalized",row[0]+".jpg")))
        for row in csv_reader:
            if row[1].count("0"):
                labels.append(0)
                images.append(image)
            elif row[1].count("1") or row[1].count("2") or row[1].count("3"):
                labels.append(1)
                images.append(image)
    return images, labels


def backtracking(idx: int, parameters: list, values: dict, chosen: dict, images: list, labels: list):
    if idx == len(parameters):
        train(chosen,images,labels)
        return
    
    for val in values[parameters[idx]]:
        chosen[parameters[idx]] = val
        backtracking(idx+1,parameters,values,chosen)
    
    return

parameters = ["nol", "nod", "af", "op", "lo"]
parameters_values = {
    "nol" : [3,4,5,6,7,8,9,10],
    "nod" : [2,3,4,5],
    "af" : ["relu", "sigmoid", "tanh"],
    "op" : ["sgd","adam","rmsprop"],
    "lo" : ["binary_crossentropy"]
}

parameters_values2 = {
    "nol" : [3],
    "nod" : [2],
    "af" : ["relu"],
    "op" : ["sgd"],
    "lo" : ["binary_crossentropy"]
}

chosen = {}
images, labels = get_dataset_0()
backtracking(0,parameters,parameters_values2,chosen,images,labels)
