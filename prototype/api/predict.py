from schemas import ImageSchema
import base64
import random

def predict_image(image):
    failure = random.randint(0,3)
    if failure == 0:
        return "sin defectos"
    elif failure == 1:
        return "sobrecarga en una fase"
    elif failure == 2:
        return "sobrecarga en dos fases"
    else:
        return "sobrecarga en tres fases"

def predict(images):
    for image in images:
        data = image.file.split(",")[1]
        bytes = base64.b64decode(data)
        image.failure = predict_image(bytes)
        image.predicted_failure = image.failure
