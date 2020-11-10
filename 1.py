import cv2 
import json
import codecs
import numpy as np
import requests
import shutil
from PIL import Image
from tqdm import tqdm
import cvlib as cv
import face_recognition
import dlib 
import os, shutil



#Load images from URL links in the JSON file to the "Images" list & store them locally to be used by the first classifier.
def LoadImages():
    imagesCount=0
    facesCount=0
    for data in tqdm(jsonData[100:103]):
        facesCount += len(data["annotation"])
        response = requests.get(data['content'], stream=True)
        with open('./temp/my_image.jpg', 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        del response
        img = Image.open('./temp/my_image.jpg')   
        img = np.asarray(img)
        cv2.imwrite('./temp/loaded_images/image{}.png'.format(imagesCount),img)
        images.append(img)
        imagesCount +=1
    return imagesCount, facesCount

#Detect Faces in the images and save their coordinates and pass them to the "DrawRectangles" function
#It's based on 4 different classifiers.
#Firstly, use the CNN library, by using APIs provided by it to detect the faces.
#If no faces were detected, then use the second classifier "face_recognition classifier"
#If no faces were detected, then use the third classifier "haarcascade classifier"
#Lastly, if no faces were predicted till now then use the last classifier which is implemented from "cvlib" library.

def FindFaces(count,imagesCount):
    count=count
    detectedFacesCount = 0
    
    for loadedImages in tqdm(range (0, imagesCount)):
        img = face_recognition.load_image_file('./temp/loaded_images/image{}.png'.format(loadedImages))
        cnn_face = dlib.cnn_face_detection_model_v1('./lib/mmod_human_face_detector.dat')
        cnn_face_detector =cnn_face(img,1)
        if len(cnn_face_detector) > 0:
            detectedFacesCount += len(cnn_face_detector)
            count = DrawRectangles(img=img, classifier=1, count=count, cnn_face_detector = cnn_face_detector)
       
        else:
            img = images[loadedImages]
            try:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                print("Something went wrong")
            finally:
                  face_locations = face_recognition.face_locations(grayscale_image)
                  if len(face_locations) > 0:
                      detectedFacesCount += len(face_locations)
                      count = DrawRectangles(img=img, classifier=2, count=count, face_locations = face_locations)
                  else:
                    face_cascade = cv2.CascadeClassifier('./lib/haarcascades/haarcascade_frontalface_alt.xml')
                    detected_faces = face_cascade.detectMultiScale(grayscale_image)
                    if len(detected_faces) > 0:
                        detectedFacesCount += len(detected_faces)
                        count = DrawRectangles(img=img, classifier=3, count=count, detected_faces = detected_faces)   
                    else:  
                        faces, confidences = cv.detect_face(img)
                        detectedFacesCount += len(faces)
                        count = DrawRectangles(img=img, classifier= 4,count=count, faces = faces, confidences=confidences)
                
    return detectedFacesCount

#Classifier numbers are {1,2,3,4} 
#1 => CNN classifier
#2 => face_detection classifier
#3 => harrcascade classifier
#4 => cvlib classifier
def DrawRectangles(img, classifier, count, detected_faces=[], faces=[], confidences=[], face_locations=[], cnn_face_detector=[] ):
    if (classifier == 2 ):
        for face_location in face_locations:
            top, right, bottom, left = face_location
    
            cv2.rectangle(
                        img,
                        (left, top),
                        (right, bottom),
                        (0, 255, 0),
                        2
                    )
    elif(classifier == 1):
        for cnn_face_detector in cnn_face_detector:
            x = cnn_face_detector.rect.left()
            y = cnn_face_detector.rect.top()
            w = cnn_face_detector.rect.right()
            h = cnn_face_detector.rect.bottom()
            cv2.rectangle(img, (x,y), (w,h), (0,0,255), 2)
    elif (classifier == 3):
        for (column, row, width, height) in detected_faces:
                cv2.rectangle(
                    img,
                    (column, row),
                    (column + width, row + height),
                    (0, 255, 0),
                    2
                )
    elif (classifier == 4):
        for face,conf in zip(faces,confidences):
                
                    (startX,startY) = face[0],face[1]
                    (endX,endY) = face[2],face[3]
                
                    # draw rectangle over face
                    cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
        
    if(classifier==2):
        cv2.imwrite('./face_detection_images/face_image_{}.jpg'.format(count),img)
    elif(classifier==1):
        cv2.imwrite('./face_detection_images/face_image_{}CNN.jpg'.format(count),img)
    elif (classifier==3):
        cv2.imwrite('./face_detection_images/face_image_{}_haarcascade.jpg'.format(count),img)  
    else:
        cv2.imwrite('./face_detection_images/face_image_{}_cvlib.jpg'.format(count),img)
    count += 1
    return count

# Starting Point of the code

address = './datasets/Face_Dataset_With_Emotion_Age_Ethnicity.json'

jsonData = []
images = []
count = 0
#Load URL links to the list jsonData & delete the url link of index (272) due to technical issues

with codecs.open(address, 'rU', 'utf-8') as js:
    for line in js:
        jsonData.append(json.loads(line))

del jsonData[119]

# delete any content in face-detection images and loaded-images
folders = ['./Face_detection_images/','./temp/loaded_images/']
for folder in folders: 
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
            
# call "LoadImages()" function & print the total number of images and faces.
imagesCount, facesCount= LoadImages()
print("\n{} images were loaded successfully, Which contains {} faces".format(imagesCount,facesCount))

# call "FindFaces()" function and print the number of faces detected using different classifiers
detectedFacesCount = FindFaces(count,imagesCount) 
print("\ndetected", detectedFacesCount, "faces")

# Program terminated successfully without any errors
accuracy= (detectedFacesCount / facesCount) * 100
print("Program terminxated successfully with accuracy: {} %".format(accuracy))