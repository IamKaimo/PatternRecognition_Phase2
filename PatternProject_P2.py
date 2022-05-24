#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import numpy as np
import time
import cv2
import os
import glob
from moviepy.editor import *


# ### Load COCO Dataset

# In[2]:


labels = open('coco.names').read().strip().split('\n')


# ### Load Neural Netowrk

# In[3]:


net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')


# ### Getting Layer Names and Output Layers

# In[4]:


names = net.getLayerNames()
outputlayers = list(net.getUnconnectedOutLayersNames())  


# ### Inference

# In[5]:


def process(image):
    boxes = []
    confidences = []
    classIDs = []
    H,W = image.shape[:2]   #height and width of the image
    blob = cv2.dnn.blobFromImage(image,1/255.0,(416,416),crop=False,swapRB=False)  #preprocessing the image
    net.setInput(blob)   #input the image to the network
    layers_output = net.forward(outputlayers)  #forward path
    
    for output in layers_output:
        for detection in output:
            scores = detection[5:]   #getting the scores of all the classes           
            classID = np.argmax(scores)  #getting ID of the class that has maximum score
            confidence = scores[classID]  #confidence of the class
            
            if confidence > 0.75:
                box = detection[:4] * np.array([W,H,W,H])  #Getting center,width and height of the box
                bx,by,bw,bh = box.astype("int")   
                x = int(bx-(bw/2))  # x-coordinate of top left corner point
                y = int(by-(bh/2))  # y-coordinate of top left corner point
                
                boxes.append([x,y,int(bw),int(bh)])  #inserting the box coordinates in boxes
                confidences.append(float(confidence)) #inserting confidence of the class in confidences
                classIDs.append(classID)  #inserting class Index in ClassIDs
                
                
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.75,0.6) #Filtering boxes using non maximum suppression
    
    
    #Using indices of filtered boxes to get their x,y,w,h
    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y = [boxes[i][0],boxes[i][1]]  # top left corner point 
            w,h = [boxes[i][2],boxes[i][3]]  #width and height of the box
            
            #Drawing boxes
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,139,139),2)
            
            #Writing the prediction on the box
            cv2.putText(image,"{}: {}".format(labels[classIDs[i]],confidences[i]),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,139,139),2)
    return image


# In[6]:


src =sys.argv[1]
dst =sys.argv[2]

clip = VideoFileClip(src)
final = clip.fl_image(process)
final.write_videofile(dst, audio=False,fps = 13)

