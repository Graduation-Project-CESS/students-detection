import cv2 
import json
import codecs
import numpy as np
import requests
import shutil
from PIL import Image
from tqdm import tqdm
import cvlib as cv
import sys

#Load images from URL links in the JSON file to the "Images" list
def LoadImages():
    imagesCount=0
    facesCount=0
    for data in tqdm(jsonData):
        
        facesCount += len(data["annotation"])
        
        response = requests.get(data['content'], stream=True)
        with open('my_image.jpg', 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        del response
        img = Image.open('my_image.jpg')   
        img = np.asarray(img)
        
        images.append(img)
        imagesCount +=1
    return imagesCount, facesCount

#Detect Faces in the images and save their coordinates and pass them to the "DrawRectangles" function
#It's based on 2 classifiers, the first one is implemented by 2 different ways
#Firstly, use the cascade classifier, if no faces were detected then resize the image to be (1280,720)
#then apply the same classifier again.
#Secondly, if no faces were detected use the API "detectfaces()" provided by the library "cvlib".

def FindFaces(count):
    count=count
    detectedFacesCount = 0
    for img in tqdm(images):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier('D:/Faculty of Engineering/ASU-4 Computer/Graduation Project/Face Detection/Face-Detection/haarcascades/haarcascade_frontalface_alt.xml')
        detected_faces = face_cascade.detectMultiScale(grayscale_image)
        if len(detected_faces) > 0:
            detectedFacesCount += len(detected_faces)
            count = DrawRectangles(img=img, classifier=1, count=count, detected_faces = detected_faces)
        else:
            img = cv2.resize(img, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
            grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('D:/Faculty of Engineering/ASU-4 Computer/Graduation Project/Face Detection/Face-Detection/haarcascades/haarcascade_frontalface_alt.xml')
            detected_faces = face_cascade.detectMultiScale(grayscale_image)
            if len(detected_faces) > 0:
                detectedFacesCount += len(detected_faces)
                count = DrawRectangles(img=img, classifier=2,count=count, detected_faces = detected_faces)
            else:
                faces, confidences = cv.detect_face(img)
                detectedFacesCount += len(faces)
                count = DrawRectangles(img=img, classifier= 3,count=count, faces = faces, confidences=confidences)
                
    return detectedFacesCount


def DrawRectangles(img, classifier, count, detected_faces=[], faces=[], confidences=[] ):
    if (classifier == 1 or classifier == 2):
        for (column, row, width, height) in detected_faces:
                cv2.rectangle(
                    img,
                    (column, row),
                    (column + width, row + height),
                    (0, 255, 0),
                    2
                )
    elif (classifier == 3):
        for face,conf in zip(faces,confidences):
                
                    (startX,startY) = face[0],face[1]
                    (endX,endY) = face[2],face[3]
                
                    # draw rectangle over face
                    cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
        
    if(classifier==1):
        cv2.imwrite('./face-detection-images/face_image_{}.jpg'.format(count),img)
    elif (classifier==2):
        cv2.imwrite('./face-detection-images/face_image_{}_resized.jpg'.format(count),img)
    else:
        cv2.imwrite('./face-detection-images/face_image_{}_cvlib.jpg'.format(count),img)
    count += 1
    return count

# Starting Point of the code

address = 'D:/Faculty of Engineering/ASU-4 Computer/Graduation Project/face_detection.json'

jsonData = []
images = []
count = 0
#Load URL links to the list jsonData & delete the url link of index (272) due to technical issues

with codecs.open(address, 'rU', 'utf-8') as js:
    for line in js:
        jsonData.append(json.loads(line))

del jsonData[272]

#call "LoadImages()" function & print the total number of images and faces.
imagesCount, facesCount= LoadImages()
print("\n",imagesCount , "images were loaded successfully, Which contains",facesCount,"faces")

#call "FindFaces()" function and print the number of faces detected using different classifiers
detectedFacesCount = FindFaces(count) 
print("\ndetected", detectedFacesCount, "faces")

#Program terminated successfully without any errors
print("done")