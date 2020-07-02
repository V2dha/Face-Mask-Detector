

!kaggle datasets download -d ashishjangra27/face-mask-12k-images-dataset

!mkdir -p ~/.kaggle

!cp kaggle\(2\).json ~/.kaggle/kaggle.json

!unzip -q /content/face-mask-12k-images-dataset.zip

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

import matplotlib.pyplot as plt

import os

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from glob import glob

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

base = MobileNetV2(input_shape = (128,128,3),include_top = False)

base = InceptionV3(input_shape = (128,128,3),include_top = False)

x = base.output
x = Flatten()(x)
x = Dense(256,activation = 'relu')(x)
x = Dropout(0.3)(x)
x = Dense(1,activation='sigmoid')(x)

model = Model(base.input,x)

model.compile('adam','binary_crossentropy',['accuracy'])

imgs = []
lbls = []

import numpy as np

!mv /content/Face\ Mask\ Dataset/Validation /content

!nvidia-smi

from tensorflow.keras.preprocessing.image import load_img,img_to_array

for i in glob('/content/Face Mask Dataset/*/WithMask/*'):
  img = load_img(i,target_size = (128,128,3))
  img = img_to_array(img)
  imgs.append(img)
  lbls.append(1)

for i in glob('/content/Face Mask Dataset/*/WithoutMask/*'):
  img = load_img(i,target_size = (128,128,3))
  img = img_to_array(img)
  imgs.append(img)
  lbls.append(0)

from sklearn.model_selection import train_test_split

imgs = np.array(imgs)
lbls = np.array(lbls)

trainx,testx,trainy,testy = train_test_split(imgs,lbls,stratify = lbls)

from tensorflow.keras.callbacks import ModelCheckpoint

mc = ModelCheckpoint('chk',save_best_only=True)

hist = model.fit(trainx,trainy,validation_data=(testx,testy),epochs = 10,batch_size = 128,callbacks=[mc])

valx = []
valy = []
for i in glob('Validation/WithMask/*'):
  img = load_img(i,target_size=(128,128,3))
  img = img_to_array(img)
  valx.append(img)
  valy.append(1)
for i in glob('Validation/WithoutMask/*'):
  img = load_img(i,target_size=(128,128,3))
  img = img_to_array(img)
  valx.append(img)
  valy.append(0)

valx = np.array(valx)
valy = np.array(valy)

model.evaluate(valx,valy,steps = 30)

import matplotlib as mpl
mpl.rcParams.update({'font.size':16})

plt.figure(figsize = (10,6))
plt.plot(hist.history['accuracy'],label = 'Train')
plt.plot(hist.history['val_accuracy'],label = 'Test')
plt.title('Training v/s Testing accuracy with successive epochs')
plt.xlabel('Epochs')
#plt.yticks((0.94,1))
plt.ylabel('Accuracy')
plt.legend()

plt.figure(figsize = (10,6))
plt.plot(hist.history['loss'],label = 'Train')
plt.plot(hist.history['val_loss'],label = 'Test')
plt.title('Training v/s Testing Loss with successive epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

from sklearn.metrics import classification_report,confusion_matrix

import seaborn as sn

cm1 = confusion_matrix(trainy,model.predict(trainx).round())

cm2 = confusion_matrix(testy,model.predict(testx).round())
cm3 = confusion_matrix(valy,model.predict(valx).round())



sn.heatmap(cm1,annot=True,fmt = '.1f',xticklabels=['No Mask','Mask'],yticklabels=['No Mask','Mask'],cmap = 'Blues')
plt.title('Train Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

sn.heatmap(cm2,annot=True,fmt = '.1f',xticklabels=['No Mask','Mask'],yticklabels=['No Mask','Mask'],cmap = 'Blues')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

sn.heatmap(cm3,annot=True,fmt = '.1f',xticklabels=['No Mask','Mask'],yticklabels=['No Mask','Mask'],cmap = 'Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

print(classification_report(trainy,model.predict(trainx).round()))

print(classification_report(testy,model.predict(testx).round()))

print(classification_report(valy,model.predict(valx).round()))

img = load_img('3.jpeg')
img = img_to_array(img)

clf = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

img = cv2.imread('1.jpeg')

boxes = clf.detectMultiScale(img,1.01,5)

boxes

img = cv2.imread('index.jpeg')
boxes = clf.detectMultiScale(img,1.01,5)
for box in boxes:
  (x,y,w,h) = box
  im = img[y:y+h,x:x+w]
  im = cv2.resize(im,(128,128))
  pred = model.predict(np.array([im]))
  if pred.round()==0:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    cv2.putText(img,'No Mask',(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
  else:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.putText(img,'Mask',(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
plt.imshow(img)

plt.imshow(img)
