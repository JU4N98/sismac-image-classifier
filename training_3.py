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

    checkpoint_callback = ModelCheckpoint(
        "./models/manual_training/aux.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

    history = compiled_model.fit(
        np.array(images_train),
        np.array(labels_train),
        shuffle=True,
        epochs=epochs,
        validation_data=(np.array(images_val), np.array(labels_val)),
        callbacks = [checkpoint_callback]
    )

    best_model = models.load_model("./models/manual_training/aux.keras")
    
    test_loss, test_accuracy = best_model.evaluate(np.array(images_test), np.array(labels_test))
    print(f"Test accuracy: {test_accuracy} - Test loss: {test_loss} ")
    
    predicted_test = best_model.predict(np.array(images_test), verbose=0)
    if model <= 2:
        predicted_test = (predicted_test >= 0.5).astype(int)
    else:
        predicted_test = np.argmax(predicted_test, axis=1)
    
    return history, labels_test, predicted_test

def show_graph(name, history, labels, predictions):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0,0].plot(loss, label="Entrenamiento")
    axes[0,0].plot(val_loss, label="Validacion")
    axes[0,0].set_title("Perdida de Entrenamiento y Validacion")
    axes[0,0].set_xlabel("Epocas")
    axes[0,0].set_ylabel("Perdida")
    axes[0,0].legend()

    axes[0,1].plot(accuracy, label="Entrenamiento")
    axes[0,1].plot(val_accuracy, label="Validacion")
    axes[0,1].set_title("Precision de Entrenamiento y Validacion")
    axes[0,1].set_xlabel("Epocas")
    axes[0,1].set_ylabel("Precision")
    axes[0,1].legend()
    axes[0,1].set_ylim(0,1.1)

    confu_matrix = confusion_matrix(labels,predictions)

    sea.heatmap(confu_matrix, annot=True, fmt="d", cmap="plasma", ax=axes[1,0])
    axes[1,0].set_title("Matriz de confusion")
    axes[1,0].set_xlabel("Etiquetas predichas")
    axes[1,0].set_ylabel("Etiquetas reales")

    class_report = classification_report(labels,predictions,output_dict=True)
    print(class_report)
    report_df = pd.DataFrame(class_report).transpose()
    class_metrics = report_df.drop(["accuracy", "macro avg", "weighted avg"])
    class_metrics.rename(columns={"precision": "Precision","recall": "Sensibilidad","f1-score": "Puntaje F1"}, inplace=True)
    bar_chart = class_metrics[["Precision", "Sensibilidad", "Puntaje F1"]].plot(kind="bar", ax=axes[1,1], colormap="plasma", rot=0)
    axes[1,1].set_title("Reporte de clasificacion por etiqueta")
    axes[1,1].set_xlabel("Etiquetas")
    axes[1,1].set_ylabel("Puntaje")
    axes[1,1].set_ylim(0, 1.1)
    axes[1,1].legend(loc="lower right")
    axes[1,1].grid(axis="y")

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

    path = "./models/manual_training/aux.png"
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
    plt.savefig(path)
    plt.show()

def clean(name):
    path = "./models/manual_training/"+name+".png"
    if os.path.exists(path):
        os.remove(path)
    
    path = "./models/manual_training/"+name+".keras"
    if os.path.exists(path):
        os.remove(path)

def save_model(name):
    shutil.copy(
        "./models/manual_training/aux.keras",
        "./models/manual_training/"+name+".keras"
    )
    shutil.copy(
        "./models/manual_training/aux.png",
        "./models/manual_training/"+name+".png"
    )

def get_best(params, model, stage, lr, epochs, name):
    finished = False

    while not finished:
        history, labels, predictions, = train(lr,epochs,model,stage,params)

        show_graph(name,history,labels,predictions)

        save = str(input("Save model (Y/N)? "))
        if save.upper() == "Y":
            clean(name)
            save_model(name)

        finished = str(input("Continue (Y/N)? ")).upper()=="N"

# MODEL 1
# First classification

# learning rate = 0.0002, epochs = 10
parameters = {"nol":4, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.9588 - loss: 0.0893 - val_accuracy: 0.9115 - val_loss: 0.3373
# Test accuracy: 0.8591549396514893 - Test loss: 0.3674169182777405
# get_best(parameters,1,1,0.0002,15,"1-1-1")

# learning rate = 0.0001, epochs = 15
parameters = {"nol":4, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.9937 - loss: 0.0818 - val_accuracy: 0.8407 - val_loss: 0.5585
# Test accuracy: 0.8450704216957092 - Test loss: 0.4858243763446808
# get_best(parameters,1,1,0.0001,15,"1-1-2")

# learning rate = 0.0001, epochs = 15
parameters = {"nol":5, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.9606 - loss: 0.1481 - val_accuracy: 0.8230 - val_loss: 0.5479
# Test accuracy: 0.8309859037399292 - Test loss: 0.4733295738697052
# get_best(parameters,1,1,0.0001,15,"1-1-3")

# Second classification

# learning rate = 0.00002, epochs = 25
parameters = {"nol":3, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.8938 - loss: 0.3383 - val_accuracy: 0.7083 - val_loss: 0.6481
# Test accuracy: 0.6333333253860474 - Test loss: 0.7052309513092041
# get_best(parameters,1,2,0.00002,25,"1-2-1")

# learning rate = 0.000005, epochs = 30
parameters = {"nol":3, "nod":3, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.8745 - loss: 0.3140 - val_accuracy: 0.6667 - val_loss: 0.7751
# Test accuracy: 0.5833333134651184 - Test loss: 0.736777126789093
# get_best(parameters,1,2,0.000005,30,"1-2-2")

# learning rate = 0.000001, epochs = 40
parameters = {"nol":3, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.8873 - loss: 0.4325 - val_accuracy: 0.7083 - val_loss: 0.6591
# Test accuracy: 0.6333333253860474 - Test loss: 0.6973279118537903
# get_best(parameters,1,2,0.000001,40,"1-2-3")

# Third classification

# learning rate = 0.00001, epochs = 20
parameters = {"nol":3, "nod":3, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.9463 - loss: 0.3261 - val_accuracy: 0.8000 - val_loss: 0.5283
# Test accuracy: 0.7419354915618896 - Test loss: 0.5519568920135498
# get_best(parameters,1,3,0.0000005,30,"1-3-1")

# learning rate = 0.000005, epochs = 50
parameters = {"nol":4, "nod":3, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
# accuracy: 0.8496 - loss: 0.4992 - val_accuracy: 0.8000 - val_loss: 0.5807
# Test accuracy: 0.774193525314331 - Test loss: 0.5893510580062866
# get_best(parameters,1,3,0.000005,25,"1-3-2")

# learning rate = 0.000003, epochs = 20
parameters = {"nol":3, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.8698 - loss: 0.3745 - val_accuracy: 0.8400 - val_loss: 0.4352
# Test accuracy: 0.774193525314331 - Test loss: 0.66941899061203
# get_best(parameters,1,3,0.000003,20,"1-3-3")

# MODEL 2
# First classification

# learning rate = 0.0001, epochs = 20
parameters = {"nol":6, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.9828 - loss: 0.0806 - val_accuracy: 0.9912 - val_loss: 0.0387
# Test accuracy: 0.98591548204422 - Test loss: 0.05162227526307106
# get_best(parameters,2,1,0.0001,20,"2-1-1")

# learning rate = 0.00013, epochs = 15
parameters = {"nol":6, "nod":3, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.9698 - loss: 0.0760 - val_accuracy: 0.9735 - val_loss: 0.0948
# Test accuracy: 0.98591548204422 - Test loss: 0.047185152769088745
# get_best(parameters,2,1,0.00013,15,"2-1-2")

# learning rate = 0.0001, epochs = 15
parameters = {"nol":7, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.9678 - loss: 0.0726 - val_accuracy: 0.9735 - val_loss: 0.1413
# Test accuracy: 0.9929577708244324 - Test loss: 0.09313762187957764
# get_best(parameters,2,1,0.0001,15,"2-1-3")

# Second classification

# learning rate = 0.000065, epochs = 30
parameters = {"nol":3, "nod":2, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.7658 - loss: 0.5162 - val_accuracy: 0.7708 - val_loss: 0.5879
# Test accuracy: 0.7833333611488342 - Test loss: 0.5252394080162048
# get_best(parameters,2,2,0.000065,60,"2-2-1")

# learning rate = 0.00005, epochs = 60
parameters = {"nol":3, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.7615 - loss: 0.4999 - val_accuracy: 0.7708 - val_loss: 0.5040
# Test accuracy: 0.699999988079071 - Test loss: 0.6140607595443726
# get_best(parameters,2,2,0.00005,30,"2-2-2")

# learning rate = 0.0004, epochs = 120
parameters = {"nol":5, "nod":3, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
# accuracy: 0.8149 - loss: 0.3959 - val_accuracy: 0.8125 - val_loss: 0.6067
# Test accuracy: 0.7666666507720947 - Test loss: 0.6400676965713501
# get_best(parameters,2,2,0.0004,120,"2-2-3")

# Third classification

# learning rate = 0.0006, epochs = 60
parameters = {"nol":3, "nod":2, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
# accuracy: 0.8536 - loss: 0.3200 - val_accuracy: 0.8000 - val_loss: 0.6142
# Test accuracy: 0.8064516186714172 - Test loss: 0.46981459856033325
# get_best(parameters,2,3,0.0006,60,"2-3-1")

# learning rate = 0.0006, epochs = 80
parameters = {"nol":3, "nod":3, "af":"relu", "op":"rmsprop", "lo":"binary_crossentropy"}
# accuracy: 0.8138 - loss: 0.3005 - val_accuracy: 0.8400 - val_loss: 0.5291
# Test accuracy: 0.7419354915618896 - Test loss: 0.563089668750762
# get_best(parameters,2,3,0.0006,80,"2-3-2")

# learning rate = 0.0002, epochs = 50
parameters = {"nol":4, "nod":4, "af":"relu", "op":"adam", "lo":"binary_crossentropy"}
# accuracy: 0.8545 - loss: 0.3495 - val_accuracy: 0.8000 - val_loss: 0.7364
# Test accuracy: 0.6774193644523621 - Test loss: 0.8462388515472412
# get_best(parameters,2,3,0.0002,50,"2-3-3")

# MODEL 3

# best val_accuracy: learning rate = 0.00001, epochs = 7
# best loss function: learning rate = 0.000003, epochs = 5
parameters={"op":"adam","lo":"sparse_categorical_crossentropy","nod":1,"nn":512,"dim":(224,224)}
# manual_training(parameters,3,0)
# accuracy: 0.9578 - loss: 0.2923 - val_accuracy: 0.6991 - val_loss: 0.8032
# Test loss: 0.7380107045173645 Test accuracy: 0.7253521084785461
# get_best(parameters,3,0,0.0000002,15,"3-0-1")

# learning rate = 0.00002, epochs = 10-15, val acc ~ 75
parameters={"op":"sgd","lo":"sparse_categorical_crossentropy","nod":1,"nn":512,"dim":(224,224)}
# manual_training(parameters,3,0)
# accuracy: 0.9562 - loss: 0.2426 - val_accuracy: 0.6991 - val_loss: 0.8470
# Test loss: 0.7674881219863892 Test accuracy: 0.7253521084785461
# get_best(parameters,3,0,0.00002,15,"3-0-2")

# learning rate = 0.00002, epochs = 10-15, val acc ~ 75
parameters={"op":"sgd","lo":"sparse_categorical_crossentropy","nod":2,"nn":512,"dim":(224,224)}
# manual_training(parameters,3,0)
# accuracy: 0.9130 - loss: 0.4402 - val_accuracy: 0.7345 - val_loss: 0.8204
# Test loss: 0.7399268746376038 Test accuracy: 0.7605633735656738
# get_best(parameters,3,0,0.00002,15,"3-0-3")

# MODEL 4

# learning rate = 0.000001, epochs = 10, val acc ~ 65
parameters={"op":"adam","lo":"sparse_categorical_crossentropy","nod":1,"nn":256,"dim":(299,299)}
# manual_training(parameters,4,0)
# accuracy: 0.9829 - loss: 0.1063 - val_accuracy: 0.6195 - val_loss: 1.7701
# Test loss: 1.9196953773498535 Test accuracy: 0.6126760840415955
# get_best(parameters,4,0,0.0000005,20,"4-0-1")

# best val_accuracy: learning rate = 0.000002, epochs = 10, val acc ~ 72
# best loss function: learning rate = 0.000001, epochs = 10, val acc ~ 65
parameters={"op":"adam","lo":"sparse_categorical_crossentropy","nod":2,"nn":512,"dim":(299,299)}
# manual_training(parameters,4,0)
# accuracy: 1.0000 - loss: 0.0237 - val_accuracy: 0.6814 - val_loss: 1.6278
# Test loss: 1.4022865295410156 Test accuracy: 0.7042253613471985
# get_best(parameters,4,0,0.000002,10,"4-0-2")

# learning rate =  0.000005, epochs = 10, val acc ~ 75
parameters={"op":"rmsprop","lo":"sparse_categorical_crossentropy","nod":1,"nn":256,"dim":(299,299)}
# manual_training(parameters,4,0)
# accuracy: 0.9666 - loss: 0.1273 - val_accuracy: 0.7168 - val_loss: 1.4690
# Test loss: 1.091317057609558 Test accuracy: 0.7464788556098938
# get_best(parameters,4,0,0.000005,10,"4-0-3")
