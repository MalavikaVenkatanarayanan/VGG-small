#from folder.filename import classname
from VGGtest.test import tinyVGG
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import cv2
import pickle
import random
matplotlib.use('Agg')
dataset='data'
model_path='model.h5'
label_path='/'
plot_path='/'
HP_LR=1e-3
HP_EPOCHS=100
HP_BS=32
HP_IMAGE_DIM=(96,96,3)
data=[]
classes=[]
imagepaths=sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagepaths)
for imgpath in imagepaths:
	try:
		image=cv2.imread(imgpath)
		image=cv2.resize(image,(96,96))
		image_array=img_to_array(image)
		data.append(image_array)
		label=imgpath.split(os.path.sep)[-2]
		classes.append(label)
	except Exception as e:
		print(e)
data=np.array(data,dtype='float')/255.0
labels=np.array(classes)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
xtrain,xtest,ytrain,ytest=train_test_split(data,labels,test_size=0.2,random_state=42)
aug = ImageDataGenerator(rotation_range=0.25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
model=tinyVGG.build(height=96,width=96,depth=3,classes=len(lb.classes_))
opt=Adam(lr=HP_LR,decay=HP_LR/HP_EPOCHS)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
H=model.fit_generator(aug.flow(xtrain,ytrain,batch_size=HP_BS),validation_data=(xtest,ytest),steps_per_epoch=len(xtrain)//HP_BS,epochs=HP_EPOCHS)
model.save('mymod.h5')