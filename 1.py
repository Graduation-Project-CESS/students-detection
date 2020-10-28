import cv2 as cv
import json
import codecs
import numpy as np
import requests
import shutil
from PIL import Image
from tqdm import tqdm

address = 'D:/Faculty of Engineering/ASU-4 Computer/Graduation Project/face_detection.json'

jsonData = []

with codecs.open(address, 'rU', 'utf-8') as js:
    for line in js:
        jsonData.append(json.loads(line))

del jsonData[272]

images = []
imagesCount=0
for data in tqdm(jsonData):
   
    response = requests.get(data['content'], stream=True)
    with open('my_image.jpg', 'wb') as file:
        shutil.copyfileobj(response.raw, file)
    del response
    img = np.asarray(Image.open('my_image.jpg'))
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    images.append(img)
    imagesCount +=1
    
print(imagesCount)

count = 0
detectedFacesCount = 0
for img in tqdm(images):

    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    face_cascade = cv.CascadeClassifier('D:/Faculty of Engineering/ASU-4 Computer/Graduation Project/Face Detection/Face-Detection/haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    for (column, row, width, height) in detected_faces:
        cv.rectangle(
            img,
            (column, row),
            (column + width, row + height),
            (0, 255, 0),
            2
        )
        detectedFacesCount +=1

    cv.imwrite('./face-detection-images/face_image_{}.jpg'.format(count),img)
    count += 1
print("detected", detectedFacesCount, "faces")

print("done")