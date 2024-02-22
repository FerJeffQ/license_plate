from ultralytics import YOLO
from PIL import Image
import easyocr 
import numpy as np
import os
from skimage import io, color, filters, transform

# obtener path actual
path = os.getcwd()

# Lector OCR
reader = easyocr.Reader(['es'])

# Modelo YOLO
detector_plate = YOLO(path + '/models/best.pt')

# Inferencia con Imagen
img = Image.open(path + "/images/carro-placa.jpg")
results = detector_plate(img)
result = results[0]

for box in result.boxes:
    label = result.names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    prob = box.conf[0].item()

    # Aplicar relleno 
    # padding = 5  # 
    # cords_padded = [max(cords[0] - padding, 0), max(cords[1] - padding, 0), min(cords[2] + padding, img.width), min(cords[3] + padding, img.height)]

    #OCR de placa
    img_crop = img.crop((cords[0], 
                         cords[1], 
                         cords[2], 
                         cords[3]))
    img_crop.save(path + '/data/crop.jpg')                         
    plate_img_np = np.array(img_crop)

    # --preprocess
 
 

    # --

    plate_text = reader.readtext(plate_img_np, detail = 0)
    print(f'License Plate Text: {plate_text}')