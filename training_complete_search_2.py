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

K_FOLD = 5
EPOCHS = 15
kfold = StratifiedKFold(n_splits=K_FOLD, shuffle=True)

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

def save_accuracy(path:str, new_row:list):
    """
    Appends a row in a csv with the model description, accuracy and validation accuracy.
    """
    print(new_row)
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_row)

def greyscale_to_rgb(images,dimensions):
    ret = []
    for image in images:
        aux = tf.image.grayscale_to_rgb(tf.convert_to_tensor(np.expand_dims(image, axis=-1)))
        aux = tf.image.resize_with_pad(aux, target_height = dimensions[0], target_width = dimensions[1])
        ret.append(aux)
    return np.array(ret)

def train_model_3(model, images_train, labels_train, images_val, labels_val, f, dim):
    checkpoint_callback = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    )

    return model.fit(
        greyscale_to_rgb(images_train[f],dim),
        np.array(labels_train[f]),
        shuffle=True,
        epochs=EPOCHS,
        validation_data=(greyscale_to_rgb(images_val[f],dim), np.array(labels_val[f])),
        callbacks = [checkpoint_callback]
    )

def test_model_3(images, labels, dim):
    best_model = models.load_model('best_model.keras')
    predictions = best_model.predict(greyscale_to_rgb(images,dim))
    predictions =  np.argmax(predictions,axis=-1)
    test_loss, test_accuracy = best_model.evaluate(greyscale_to_rgb(images,dim), np.array(labels))
    return [test_accuracy, test_loss]

def train(params: dict, images: list, labels: list, create_model,train_model,test_model,path:str):
    """
    Trains and evaluates the model through each fold.
    """
    images_test = []
    labels_test = []
    images, images_test, labels, labels_test = train_test_split(images,labels,test_size=0.2,stratify=labels)

    # Creates folds and balances labels
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
        history = train_model(model,images_train,labels_train,images_val,labels_val,f,params["dim"])
        row = [str(params),history.history["accuracy"],history.history["val_accuracy"],history.history["loss"],history.history["val_loss"]]
        row.extend(test_model(images_test, labels_test, params["dim"]))
        save_accuracy(path,row)

def get_dataset_3(path:str):
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
            elif row[1].count("1"):
                labels.append(1)
                images.append(image)
            elif row[1].count("2"):
                labels.append(2)
                images.append(image)
            elif row[1].count("3"):
                labels.append(3)
                images.append(image)
    return images, labels

def create_model_3(params: dict):
    dim = params["dim"]
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(dim[0], dim[1], 3))

    # freezes all layers in the model
    for layer in model.layers:
        layer.trainable = False

    # adds custom classification layers
    output = model.output
    output = layers.Flatten()(output) 
    # output = layers.Dropout(0.5)(output)
    for i in range(params["nod"]):
        output = layers.Dense(params["nn"], activation='relu')(output)
    predictions = layers.Dense(4, activation='softmax')(output) 

    model = models.Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=params["op"], 
              loss=params["lo"], 
              metrics=['accuracy'])
    
    model.summary()
    
    return model

def create_model_4(params: dict):
    dim = params["dim"]
    model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(dim[0], dim[1], 3))

    # freezes all layers in the model
    for layer in model.layers:
        layer.trainable = False

    # adds custom classification layers
    output = model.output
    output = layers.Flatten()(output) 
    # output = layers.Dropout(0.5)(output)
    output = layers.Dense(params["nn"], activation='relu')(output)
    predictions = layers.Dense(4, activation='softmax')(output) 

    model = models.Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=params["op"], 
              loss=params["lo"], 
              metrics=['accuracy'])
    
    model.summary()

    return model

# def create_model_5(params: dict):
#     dim = params["dim"]
#     model =  tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(dim[0], dim[1], 3))

#     # freezes all layers in the model
#     for layer in model.layers:
#         layer.trainable = False

#     # adds custom classification layers
#     output = model.output
#     output = layers.Flatten()(output) 
#     # output = layers.Dropout(0.5)(output)
#     output = layers.Dense(params["nn"], activation='relu')(output)
#     predictions = layers.Dense(4, activation='softmax')(output) 

#     model = models.Model(inputs=model.input, outputs=predictions)
#     model.compile(optimizer=params["op"], 
#               loss=params["lo"], 
#               metrics=['accuracy'])
    
#     model.summary()

#     return model

def complete_search(idx: int, parameters: list, values: dict, chosen: dict, images: list, labels: list, create_model, train_model, test_model, path:str):
    """
    Evaluates each of the models that can be get from all combinations of parameters values.
    """
    if idx == len(parameters):
        train(chosen,images,labels,create_model,train_model,test_model,path)
        return
    
    for val in values[parameters[idx]]:
        chosen[parameters[idx]] = val
        complete_search(idx+1,parameters,values,chosen,images,labels,create_model,train_model,test_model,path)
    
    return

# Model 3: Resnet fine-tunning 

parameters = ["op", "lo","nod","nn","dim"]
parameters_values = {
    "op" : ["sgd","adam","rmsprop"], # optimizer
    "lo" : ["sparse_categorical_crossentropy"], # loss function
    "nod": [1,2], # number of dense layers
    "nn" : [256,512], # number of neurons in dense layer
    "dim": [(224,224)] # dimensions of the input model
}

chosen = {}
images, labels = get_dataset_3("./dataset_normalized_2")
complete_search(0,parameters,parameters_values,chosen,images,labels,create_model_3,train_model_3,test_model_3,"./results/complete_search/3.csv")

# Model 4: Inception fine-tunning

parameters = ["op", "lo","nod","nn","dim"]
parameters_values = {
    "op" : ["sgd", "adam", "rmsprop"], # optimizer
    "lo" : ["sparse_categorical_crossentropy"], # loss function
    "nod": [1,2], # number of dense layers
    "nn" : [256,512], # number of neurons in dense layer
    "dim": [(299,299)] # dimensions of the input model
}

chosen = {}
images, labels = get_dataset_3("./dataset_normalized_2")
complete_search(0,parameters,parameters_values,chosen,images,labels,create_model_4,train_model_3,test_model_3,"./results/complete_search/4.csv")

# Model 5: VGG fine-tunning

# parameters = ["op", "lo","nn","dim"]
# parameters_values = {
#     "op" : ["sgd","adam","rmsprop"],
#     "lo" : ["sparse_categorical_crossentropy"],
#     "nn" : [256,512,1024],
#     "dim": [(224,224)]
# }

# chosen = {}
# images, labels = get_dataset_3("./dataset_normalized_2")
# backtracking(0,parameters,parameters_values,chosen,images,labels,create_model_5,train_model_3,"./results/complete_search/5.csv")
