import os 
from pathlib import Path
import json
import pytesseract
from PIL import Image
import csv

MIN_T = 0
MAX_T = 300
pytesseract.pytesseract.tesseract_cmd = os.path.join(Path.home(),"AppData/Local/Programs/Tesseract-OCR/tesseract.exe")

def rename_images():
    dirname = "./dataset_raw"
    for i, filename in enumerate(os.listdir(dirname)):
        os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".jpg")

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

def get_mapping() -> dict:
    """
        Returns a dictionary that maps every rgb value to a
        greyscale value [0,255].
    """

    # try to load mapping from file
    try:
        with open("rgb_to_grey.json") as json_file:
            return json.load(json_file)
    except:
        print("rgb_to_grey.json not found, calculating it ...")

    sample = Image.open("./dataset/sample.jpg")
    image = sample.load()

    # get right-side scale
    t_scale = []
    for i in range(49,271): t_scale.append(image[237,i])
    t_scale = list(reversed(t_scale))

    # assign value between 0 and 255 to every pixel on right-side scale
    grey_scale = []
    for i in range(0,222):
        grey_scale.append(i*255//221)
    
    # map every combination in rgb scale to a greyscale value
    rgb_to_grey = {}
    for r in range(0,256):
        for g in range(0,256):
            for b in range(0,256):
                idx = -1
                distance = 255*3 + 1
                for i in range(0,222):
                    if(get_distance(t_scale[i],(r,g,b)) < distance):
                        idx = i
                        distance = get_distance(t_scale[i],(r,g,b))
                rgb_to_grey[get_rgb_mask((r,g,b))] = grey_scale[idx];

    with open("rgb_to_grey.json", "w") as json_file:
        json.dump(rgb_to_grey, json_file)
    
    return rgb_to_grey

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

def to_greyscale(path:str, filename: str, rgb_to_grey:dict, min_t:float, max_t:float):
    """
        Generates greyscale image in absolute scale from rgb one.
    """
    
    file = Image.open(os.path.join(path,filename))
    image = file.load()

    # temperatures = get_temperatures(image)
    # if len(temperatures) != 2:
    #     return

    # min_t = temperatures[0]
    # max_t = temperatures[1]

    greyscale_image = []
    for r in range (33,292):
        for c in range (0,234):
            cur_t = rgb_to_grey[get_rgb_mask(image[c,r])]*int(max_t-min_t) // 255 + int(min_t)
            greyscale_image.append(cur_t * 255 // MAX_T)
    
    new_image = Image.new("L",(234,259))
    new_image.putdata(greyscale_image)
    new_image.save("./dataset_normalized/"+filename)

def normalize_all():
    rgb_to_grey = get_mapping()
    csv_file = "./dataset/temperatures.csv"
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            to_greyscale("./dataset_raw",row[0]+".jpg",rgb_to_grey,float(row[1]),float(row[2]))

normalize_all()
