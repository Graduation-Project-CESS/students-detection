import cv2 
#import json
#import codecs
#import numpy as np
#import requests
#import shutil
import glob
#import tensorflow
#from PIL import Image
#from tqdm import tqdm
import cvlib as cv
import face_recognition

#Load images from URL links in the JSON file to the "Images" list & store them locally to be used by the first classifier.
#def LoadImages():
  #  imagesCount=0
  #  facesCount=0
   # for data in tqdm(jsonData[0:4]):
    #    facesCount += len(data["annotation"])
     #   response = requests.get(data['content'], stream=True)
      #  with open('my_image.jpg', 'wb') as file:
       #     shutil.copyfileobj(response.raw, file)
        #del response
        #img = Image.open('my_image.jpg')   
        #img = np.asarray(img)
        #cv2.imwrite('./loaded_images/image{}.png'.format(imagesCount),img)
        #images.append(img)
        #imagesCount +=1
    #return imagesCount, facesCount

#Detect Faces in the images and save their coordinates and pass them to the "DrawRectangles" function
#It's based on 3 different classifiers.
#Firstly, use the face_detection library, by using APIs provided by it to detect the faces.
#If no faces were detected, then use the second classifier "haarcascade classifier"
#Lastly, if no faces were predicted till now then use the last classifier which is implemented from "cvlib" library.

def FindFaces(count,imagesCount):
    count=count
    detectedFacesCount = 0
    
    for loadedImages in range (0,imagesCount):
        img = face_recognition.load_image_file('D:/Mayar/ASU CHEP smester9/grad/Face-Detection/loaded_images/image{}.png'.format(loadedImages))
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) > 0:
            detectedFacesCount += len(face_locations)
            count = DrawRectangles(img=img, classifier=1, count=count, face_locations = face_locations)
        else:
            img = images[loadedImages]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            face_cascade = cv2.CascadeClassifier('D:/Mayar/ASU CHEP smester9/grad/Face-Detection/haarcascades/haarcascade_frontalface_alt.xml')
            detected_faces = face_cascade.detectMultiScale(grayscale_image)
            if len(detected_faces) > 0:
                detectedFacesCount += len(detected_faces)
                count = DrawRectangles(img=img, classifier=2, count=count, detected_faces = detected_faces)
            else:
                faces, confidences = cv.detect_face(img)
                detectedFacesCount += len(faces)
                count = DrawRectangles(img=img, classifier= 3,count=count, faces = faces, confidences=confidences)
                
    return detectedFacesCount

#Classifier numbers are {1,2,3} 
#1 => face_detection classifier
#2 => harrcascade classifier
#3 => cvlib classifier
def DrawRectangles(img, classifier, count, detected_faces=[], faces=[], confidences=[], face_locations=[] ):
    if (classifier == 1 ):
        for face_location in face_locations:
            top, right, bottom, left = face_location
    
            cv2.rectangle(
                        img,
                        (left, top),
                        (right, bottom),
                        (0, 255, 0),
                        2
                    )
            
    elif (classifier == 2):
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
        cv2.imwrite('D:/Mayar/ASU CHEP smester9/grad/Face-Detection/Face_detection_images/face_image_{}.jpg'.format(count),img)
    elif (classifier==2):
        cv2.imwrite('D:/Mayar/ASU CHEP smester9/grad/Face-Detection/Face_detection_images/face_image_{}_haarcascade.jpg'.format(count),img)
    else:
        cv2.imwrite('D:/Mayar/ASU CHEP smester9/grad/Face-Detection/Face_detection_images/face_image_{}_cvlib.jpg'.format(count),img)
    count += 1
    return count

# Starting Point of the code
images = []
count = 0
imagesCount = facesCount = 960
path = glob.glob("D:/Mayar/ASU CHEP smester9/grad/real_and_fake_face_detection/real_and_fake_face/training_fake/*.jpg")
imagesCount=0

for img in path:
    
    im= cv2.imread(img)
    images.append(im)
    cv2.imwrite('D:/Mayar/ASU CHEP smester9/grad/Face-Detection//loaded_images/image{}.png'.format(imagesCount),im)
    imagesCount+=1
    
    
    
#jsonData = []

#Load URL links to the list jsonData & delete the url link of index (272) due to technical issues

#with codecs.open(address, 'rU', 'utf-8') as js:
   # for line in js:
        #jsonData.append(json.loads(line))

#del jsonData[272]

#call "LoadImages()" function & print the total number of images and faces.
#imagesCount, facesCount= LoadImages()
#print("\n{} images were loaded successfully, Which contains {} faces".format(imagesCount,facesCount))

#call "FindFaces()" function and print the number of faces detected using different classifiers
detectedFacesCount = FindFaces(count,imagesCount) 
print("\n detected", detectedFacesCount, "faces")

#Program terminated successfully without any errors
accuracy= (detectedFacesCount / facesCount) * 100
print("Program terminated successfully with accuracy: {} %".format(accuracy))