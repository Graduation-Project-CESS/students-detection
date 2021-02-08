import cv2 
# import json
# import codecs
# import numpy as np
# import requests
import shutil
import glob
# from PIL import Image
from tqdm import tqdm
import cvlib as cv
import face_recognition
import dlib 
import os
import time
from numba import vectorize, jit, cuda 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
#Load images from URL links in the JSON file to the "Images" list & store them locally to be used by the first classifier.
def LoadImages():
    imagesCount=0
    facesCount=0
    for data in tqdm(jsonData[20:30]):
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
'''
'''
Detect Faces in the images and save their coordinates in detectedFacesLoc
pass the image to the 3 classifiers sequentially CNN then Face-recognition then CVLib
It's based on 3 different classifiers.
Firstly, use the CNN library, by using APIs provided by it to detect the faces.
then use the second classifier "face_recognition classifier"
Lastly use the last classifier which is implemented from "cvlib" library.

after running first classifier and savig coordinates of all faces detected
we run second classifier and compare all detected faces by second classifier to
the ones by first classifier and if any 2 faces have same coordinates range 
we save one of them but we increase its count(no. of times this face is found)

if there is a face found only 1 time it is not saved
we save only faces that are found more than one time
'''
#@jit(nopython=True)
#@vectorize(['float64(float64)'], target ="cuda")                         
#@jit(target ="cuda")                          
def FindFaces(imagesCount):
    
    detectedFacesCount = 0
    detectedFacesLoc = []  
    cnn_face = dlib.cnn_face_detection_model_v1('./lib/mmod_human_face_detector.dat')
    for loadedImages in tqdm(range (0, imagesCount)):
        
        img = face_recognition.load_image_file('./temp/loaded_images/image{}.png'.format(loadedImages))
        h,w,c = img.shape
        start_time=time.time()
        
        # if h > 1280 and w > 720:
        #     img=cv2.resize(img, dsize=(1280,720), interpolation=cv2.INTER_CUBIC)
        face_locations =cnn_face(img,0)
        print(len(face_locations))
        if (len(face_locations) == 0):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("upscaling")
            face_locations =cnn_face(img,1)
        end_time=time.time()
        print("time : ", end_time - start_time, "sec")
        if (len(face_locations) > 0):  
            for face in face_locations:
                x1 = face.rect.left()
                y1 = face.rect.top()
                x2 = face.rect.right()
                y2 = face.rect.bottom()
                c = 1
                count = 1
                detectedFacesLoc.append([x1,y1,x2,y2,c,count])
                print("\nCNN:\n")
                print([x1,y1,x2,y2,c,count])


        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            print("Something went wrong")
        finally:
            face_locations = face_recognition.face_locations(grayscale_image)
        
        if (len(face_locations) > 0):  
            for face in face_locations:
                y1,x2,y2,x1 = face
                found = 0
                for dface in detectedFacesLoc:
                    if (x1 < dface[0] + 30 and x1 > dface[0] - 30):
                        found = 1
                        print("\nfr:\n")
                        print(dface)
                        dface[5] += 1
                        print(dface)
                        break
                if (found == 0):
                    c = 2
                    detectedFacesLoc.append([x1,y1,x2,y2,c,1])
         
                     
        face_locations, confidences = cv.detect_face(img)
        if (len(face_locations) > 0):  
            for face in face_locations:
                x1,y1,x2,y2 = face
                found = 0
                for dface in detectedFacesLoc:
                    if (x1 < dface[0] + 30 and x1 > dface[0] - 30):
                        found = 1
                        print("\ncvlib:\n")
                        print(dface)
                        dface[5] += 1
                        print(dface)
                        break
                if (found == 0):
                    c = 3
                    detectedFacesLoc.append([x1,y1,x2,y2,c,1])
                    
            
        for f in detectedFacesLoc:
            if (f[5] > 1):              
                if (f[4] == 1) :
                    cv2.rectangle(img, (f[0],f[1]), (f[2],f[3]), (255,0,0), 2)
                elif (f[4] == 2) :
                   cv2.rectangle(img, (f[0],f[1]), (f[2],f[3]), (0,255,0), 2)                
                else:
                   cv2.rectangle(img, (f[0],f[1]), (f[2],f[3]), (0,0,255), 2)
            else:
                print("\nrubbish:\n")
                print(f)
                detectedFacesLoc.remove(f)
            
        cv2.imwrite('./face_detection_images/face_image_{}.jpg'.format(loadedImages),img)
        detectedFacesCount += len(detectedFacesLoc)
        print("\nfound:\n")
        print(detectedFacesLoc)
        print("\n")
        detectedFacesLoc = []
    return detectedFacesCount


# Starting Point of the code

address = './datasets/1.json'

jsonData = []
images = []
'''
#Load URL links to the list jsonData & delete the url link of index (272) due to technical issues
with codecs.open(address, 'rU', 'utf-8') as js:
    for line in js:
        jsonData.append(json.loads(line))

del jsonData[119]


# call "LoadImages()" function & print the total number of images and faces.
imagesCount, facesCount= LoadImages()
print("\n{} images were loaded successfully, Which contains {} faces".format(imagesCount,facesCount))

'''      
# dlib.DLIB_USE_CUDA=True
# print("sadasdasdasdasd",dlib.DLIB_USE_CUDA)

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
            
            
path = glob.glob("./images/*.jpg")
imagesCount=0

for img in tqdm(path):
    im= cv2.imread(img)
    h,w,c = im.shape
    if h > 1280 and w > 720:
        im=cv2.resize(im, dsize=(1280,720), interpolation=cv2.INTER_CUBIC)
    images.append(im)
    cv2.imwrite('./temp/loaded_images/image{}.png'.format(imagesCount),im)
    imagesCount+=1
print("\nloaded images successfully")
          


# call "FindFaces()" function and print the number of faces detected using different classifiers
detectedFacesCount = FindFaces(imagesCount) 
print("\ndetected", detectedFacesCount, "faces")

# Program terminated successfully without any errors
#accuracy= (detectedFacesCount / facesCount) * 100
#print("Program terminxated successfully with accuracy: {} %".format(accuracy))