import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from pathlib import Path
from io import BytesIO
import json
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import pytesseract

MIN_T = 0
MAX_T = 300
pytesseract.pytesseract.tesseract_cmd = os.path.join(Path.home(),"AppData/Local/Programs/Tesseract-OCR/tesseract.exe")

def get_mapping() -> dict:
    """
        Returns a dictionary that maps every rgb value to a
        greyscale value [0,255].
    """
    with open("rgb_to_grey.json") as json_file:
        return json.load(json_file)
    
rgb_to_grey = get_mapping()

def get_distance(pixel_1:tuple, pixel_2:tuple) -> int:
    """
        Returns the Manhattan distance between two rgb values.
    """
    return abs(pixel_1[0] - pixel_2[0]) + abs(pixel_1[1] - pixel_2[1]) + abs(pixel_1[2] - pixel_2[2])

def get_rgb_mask(rgb: tuple) -> str:
    """
        Returns the rgb mask of a rgb tuple.
    """
    return str(rgb[0]*(256**2) + rgb[1] * 256 + rgb[2])

def get_temperatures(image) -> list:
    """
        Returns minimum and maximum temperature from image.
    """
    cropped_image = []
    for r in range(290,305):
        for c in range(0,200):
            if get_distance(image[c,r],(255,255,255))<250:
                cropped_image.append((0,0,0))
            else:
                cropped_image.append((255,255,255))
    
    new_image = Image.new("RGB", (200,15))
    new_image.putdata(cropped_image)

    recognized = pytesseract.image_to_string(new_image, lang='eng', config='--psm 7')
    temperatures = []
    i : int = 0
    while i < len(recognized):
        cur = ""
        while i<len(recognized) and (recognized[i].isdigit() or recognized[i]=='.'):
            cur += recognized[i]
            i += 1
        if cur != "": 
            temperatures.append(float(cur))
            i -= 1
        i += 1

    return sorted(temperatures)

def to_greyscale(bytes: bytes, rgb_to_grey:dict):
    """
        Generates greyscale image in absolute scale from rgb one.
    """
    file = Image.open(BytesIO(bytes))
    image = file.load()

    temperatures = get_temperatures(image)
    if len(temperatures) != 2:
        return
    min_t = temperatures[0]
    max_t = temperatures[1]

    greyscale_image = []
    for r in range (33,292):
        for c in range (0,234):
            cur_t = rgb_to_grey[get_rgb_mask(image[c,r])]*int(max_t-min_t) // 255 + int(min_t)
            greyscale_image.append(cur_t * 255 // MAX_T)
    
    new_image = Image.new("L",(234,259))
    new_image.putdata(greyscale_image)
    return new_image

def downscale(image:Image):
    return image.resize((78,86), Image.LANCZOS)

def get_histogram(image: Image):
    return np.histogram(image, bins=256, range=(0, 255), density=True)[0]

def add_padding(image:Image, dim:tuple):
    return tf.image.resize_with_pad(image, target_height=dim[0], target_width=dim[1])

class Model2():
    def __init__(self, paths):
        self.models = [tf.keras.models.load_model(path) for path in paths]

    def normalize(self, bytes: bytes):
        image = to_greyscale(bytes, rgb_to_grey)
        return get_histogram(image)
    
    def predict(self, bytes):
        image = self.normalize(bytes)
        print(image)
        idx_to_label = [0,3,1]
        for idx, model in enumerate(self.models):
            prediction = model.predict(np.array([image]), verbose=0)[0][0]
            if prediction < 0.5:
                return idx_to_label[idx]
        return 2

model = Model2(["./models/2-1-1.keras", "./models/2-2-1.keras", "./models/2-3-1.keras"])\

def predict(images):
    for image in images:
        data = image.file.split(",")[1]
        bytes = base64.b64decode(data)
        prediction = model.predict(bytes)
        failure = ""
        if prediction == 0:
            failure = "sin defectos"
        elif prediction == 1:
            failure = "sobrecarga en una fase"
        elif prediction == 2:
            failure = "sobrecarga en dos fases"
        else:
            failure = "sobrecarga en tres fases"
        image.failure = failure
        image.predicted_failure = failure
