from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import easyocr
import numpy as np
import os

# Crear un lector de OCR en español
reader = easyocr.Reader(['es'])

#Obtener el directorio actual
path = os.getcwd()

# Cargar el modelo YOLO
license_plate_detector = YOLO(path + "/models/best.pt")

# Realizar la inferencia en la imagen
results = license_plate_detector(path + "/images/carro-placa.jpg")

result = results[0]

# Leer la imagen original
img = Image.open(path + "/images/carro-placa.jpg")

# Mostrar la imagen
plt.imshow(img)
plt.axis('off')

# Iterar sobre los resultados y dibujar los cuadros delimitadores
for box in result.boxes:
    label = result.names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    prob = box.conf[0].item()
    
    # Si el objeto detectado es una placa de vehículo, aplicar OCR
    if label == 'license_plate':
        # Extraer la región de la imagen correspondiente a la placa
        license_plate_img = img.crop((cords[0], cords[1], cords[2], cords[3]))
        
        # Convertir la imagen de la placa a un array de numpy
        license_plate_img_np = np.array(license_plate_img)
        
        # Aplicar OCR a la imagen de la placa
        license_plate_text = reader.readtext(license_plate_img_np, detail = 0)
        
        # Imprimir el texto reconocido
        print(f'License Plate Text: {license_plate_text}')
    
    # Dibujar el cuadro delimitador
    rect = patches.Rectangle((cords[0], cords[1]), cords[2] - cords[0], cords[3] - cords[1], linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    
    # Mostrar la etiqueta y la probabilidad
    plt.text(cords[0], cords[1], f'{label}: {prob:.2f}', bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')

# Mostrar la imagen con los cuadros delimitadores y etiquetas
plt.show()