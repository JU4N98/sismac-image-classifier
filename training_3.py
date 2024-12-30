import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import csv
import numpy as np
from PIL import Image 
import tensorflow as tf, keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from random import randint
import matplotlib.pyplot as plt

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

def create_model_3(params: dict, lr:float):
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Freezes all layers in the model
    for layer in model.layers:
        layer.trainable = False
    # Adds flattening layer
    output = model.output
    output = layers.Flatten()(output) 
    # Adds dense layers
    for i in range(params["nod"]):
        output = layers.Dense(params["nn"], activation='relu')(output)
    predictions = layers.Dense(4, activation='softmax')(output) 
    # Compiles model
    model = models.Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=get_optimizer(params["op"],lr), 
              loss=params["lo"], 
              metrics=['accuracy'])
    return model

def create_model_4(params: dict, lr:float):
    model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    # Freezes all layers in the model
    for layer in model.layers:
        layer.trainable = False
    # Adds flattening layer
    output = model.output
    output = layers.Flatten()(output) 
    # Adds dense layers
    output = layers.Dense(params["nn"], activation='relu')(output)
    predictions = layers.Dense(4, activation='softmax')(output) 
    # Compiles model
    model = models.Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=get_optimizer(params["op"],lr), 
              loss=params["lo"], 
              metrics=['accuracy'])
    return model

def get_model(lr, model, stage, params):
    compiled_model = None

    if model == 1:
        compiled_model = create_model_1(params,lr)
    elif model == 2:
        compiled_model = create_model_2(params,lr)
    elif model == 3:
        compiled_model = create_model_3(params,lr)
    elif model == 4:
        compiled_model = create_model_4(params,lr)
    
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

def get_dataset(model, stage):
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

    get_label = {
        1: {
            1:{ "0":0, "1":1, "2":1, "3":1, },
            2:{ "1":1, "2":1, "3":0, },
            3:{ "1":0, "2":1 },
        },
        2:{
            1:{ "0":0, "1":1, "2":1, "3":1, },
            2:{ "1":1, "2":1, "3":0, },
            3:{ "1":0, "2":1, },
        },
        3: {"0":0,"1":1,"2":2,"3":3,},
        4: {"0":0,"1":1,"2":2,"3":3,},
    }

    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            image = np.array(Image.open(os.path.join(dataset_path,row[0]+".jpg")))
            
            get_label_2 = {}
            if model == 1 or model == 2:
                get_label_2  = get_label[model][stage]
            else:
                get_label_2 = get_label[model]
            
            added = False
            for key, val in get_label_2.items():
                if row[1].count(key):
                    labels.append(val)
                    added = True
            if not added:
                continue

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

def train(lr, epochs, model, stage, params):
    images, labels = get_dataset(model,stage)
    images_test = []
    labels_test = []
    images, images_test, labels, labels_test = train_test_split(images,labels,test_size=0.2,stratify=labels)
    images_train, images_val, labels_train, labels_val = train_test_split(images,labels,test_size=0.2,stratify=labels)

    maximum = 0
    images_aux = []
    labels_aux = []
    for key, value in split(images_train, labels_train).items():
        maximum = max(maximum, len(value))
    for key, value in split(images_train, labels_train).items():
        images_to_add = value
        labels_to_add = [key]*len(value)
        oversampling(images_to_add,labels_to_add,maximum)
        images_aux.extend(images_to_add)
        labels_aux.extend(labels_to_add)
    images_train = images_aux
    labels_train = labels_aux

    compiled_model = get_model(lr,model,stage,params)

    return compiled_model.fit(
        np.array(images_train),
        np.array(labels_train),
        shuffle=True,
        epochs=epochs,
        validation_data=(np.array(images_val), np.array(labels_val)),
    )

def show_graph(accuracy, val_accuracy, loss, val_loss):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim(0,1.1)

    plt.show()

def manual_training(params, model, stage):
    epochs = 15
    initial_lrs = {"adam":0.001,"sgd":0.01,"rmsprop":0.01}
    lr = initial_lrs[params["op"]]

    while True:
        print(f"Current learning rate = {lr}, Current # of epochs = {epochs}")
        lr_nxt = float(input("Learining rate: "))
        epoch_nxt = int(input("Epochs: "))

        epochs = epoch_nxt
        lr = lr_nxt

        history = train(lr,epochs,model,stage,params)
        show_graph(history.history["accuracy"],history.history["val_accuracy"],history.history["loss"],history.history["val_loss"])

# MODEL 1
# First classification

# learning rate = 0.0006, epochs = 15
parameters = {"nol":4, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,1,1)

# learning rate = 0.0008, epochs = 15
parameters = {"nol":4, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,1,1)

# learning rate = 0.0008, epochs = 15
parameters = {"nol":5, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,1,1)

# Second classification

# learning rate = 0.0009, epochs = 15
parameters = {"nol":3, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,1,2)

# learning rate = 0.0001, epochs = 15
parameters = {"nol":3, "nod":3, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,1,2)

# learning rate = 0.00005, epochs = 25
parameters = {"nol":3, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,1,2)

# Third classification

# learning rate = 0.00001, epochs = 20
parameters = {"nol":3, "nod":3, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,1,3)

# learning rate = 0.000005, epochs = 50
parameters = {"nol":4, "nod":3, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
# manual_training(parameters,1,3)

# learning rate = 0.00001, epochs = 20
parameters = {"nol":3, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,1,3)

# MODEL 2
# First classification

# learning rate = 0.0001, epochs = 20
parameters = {"nol":6, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,2,1)

# learning rate = 0.00013, epochs = 15
parameters = {"nol":6, "nod":3, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,2,1)

# learning rate = 0.0001, epochs = 15
parameters = {"nol":7, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,2,1)

# Second classification

# learning rate = 0.000065, epochs = 30
parameters = {"nol":3, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,2,2)

# learning rate = 0.00005, epochs = 60
parameters = {"nol":3, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,2,2)

# learning rate = 0.0004, epochs = 120
parameters = {"nol":5, "nod":3, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
# manual_training(parameters,2,2)

# Third classification

# learning rate = 0.0006, epochs = 60
parameters = {"nol":3, "nod":2, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
# manual_training(parameters,2,3)

# learning rate = 0.0006, epochs = 80
parameters = {"nol":3, "nod":3, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
# manual_training(parameters,2,3)

# learning rate = 0.001, epochs = 50
parameters = {"nol":4, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# manual_training(parameters,2,3)

# MODEL 3

# best val_accuracy: learning rate = 0.00001, epochs = 7
# best loss function: learning rate = 0.000003, epochs = 5
parameters={"op":"adam","lo":"sparse_categorical_crossentropy","nod":1,"nn":512,"dim":(224,224)}
# manual_training(parameters,3,0)

# learning rate = 0.00002, epochs = 10-15, val acc ~ 75
parameters={"op":"sgd","lo":"sparse_categorical_crossentropy","nod":1,"nn":512,"dim":(224,224)}
# manual_training(parameters,3,0)

# learning rate = 0.00002, epochs = 10-15, val acc ~ 75
parameters={"op":"sgd","lo":"sparse_categorical_crossentropy","nod":2,"nn":512,"dim":(224,224)}
# manual_training(parameters,3,0)

# MODEL 4

# learning rate = 0.000001, epochs = 10, val acc ~ 65
parameters={"op":"adam","lo":"sparse_categorical_crossentropy","nod":1,"nn":256,"dim":(299,299)}
# manual_training(parameters,4,0)

# best val_accuracy: learning rate = 0.000002, epochs = 10, val acc ~ 72
# best loss function: learning rate = 0.000001, epochs = 10, val acc ~ 65
parameters={"op":"adam","lo":"sparse_categorical_crossentropy","nod":2,"nn":512,"dim":(299,299)}
# manual_training(parameters,4,0)

# learning rate =  0.00001, epochs = 10, val acc ~ 75
parameters={"op":"rmsprop","lo":"sparse_categorical_crossentropy","nod":1,"nn":256,"dim":(299,299)}
# manual_training(parameters,4,0)
