from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
from ultralytics import YOLO
from PIL import Image
import easyocr 
import numpy as np
import os




# obtener path actual
path = os.getcwd()

# Lector OCR
reader = easyocr.Reader(['es'])

# Modelo YOLO
detector_plate = YOLO(path + '/models/best.pt')


# Create video capture object
capture = VideoCapture('rtsp://fernando:ubuntu@192.168.1.43:3002/h264_pcm.sdp')
 
# Check that a camera connection has been established
if not capture.isOpened():
    print("Error establishing connection")
 
while capture.isOpened():
 
    # Read an image frame
    ret, img = capture.read()

    results = detector_plate(img)
    result = results[0]
    
    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        cords = [round(x) for x in box.xyxy[0].tolist()]
        prob = box.conf[0].item()

        #OCR de placa
        img_crop = img[cords[1]:cords[3], cords[0]:cords[2]]

        # img_crop = img.crop((cords[0], 
        #                     cords[1], 
        #                     cords[2], 
        #                     cords[3]))
        #img_crop.save(path + '/data/crop.jpg')                         
        plate_img_np = np.array(img_crop)

        # --preprocess
    
    

        # --

        plate_text = reader.readtext(plate_img_np, detail = 0)
        print(f'License Plate Text: {plate_text}')


    # If an image frame has been grabbed, display it
#    if ret:
#        imshow('Displaying image frames from a webcam', frame)
 
    # If the Esc key is pressed, terminate the while loop
#    if waitKey(25) == 27:
#        break
 
# Release the video capture and close the display window
#capture.release()
#destroyAllWindows()