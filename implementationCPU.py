from IPython.display import Image
import tensorflow
print(tensorflow.__version__)
import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import np_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import glob
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np


datasetRoot='C://Users//muniv//Desktop//Code4fun//1_originals//vgg16_pytorch//garbage'
classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
            'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
nbClasses = len(classes)
itemPerClass = 100
N = nbClasses * itemPerClass
npix =  224


classLabel = 0

x = np.empty(shape=(0,npix,npix,3))
y = []
first = True
for cl in classes:
    listImages = glob.glob(datasetRoot + '/' + cl + '/*')
    y += [classLabel]*itemPerClass
    for pathImg in tqdm(listImages[:itemPerClass]):
        img = image.load_img(pathImg, target_size=(npix, npix))
        im = image.img_to_array(img)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        x = np.vstack([x, im])
    classLabel += 1
y = keras.utils.np_utils.to_categorical(y, nbClasses)

#Shuffling the image array
#shuffle imported data and divide train and test sets
from random import shuffle
import numpy as np

y = np.array(y)
ind_list = [i for i in range(N)]
shuffle(ind_list)
xNew = x[ind_list, :,:,:]
yNew = y[ind_list,]

pTrain = int(0.8*N) #you may change here for a different ratio train/test
xTrain = xNew[:pTrain]
xTest  = xNew[pTrain:]

yTrain = yNew[:pTrain]
yTest  = yNew[pTrain:]


#loading machine learning model 
VGGmodel = VGG16(weights='imagenet', include_top=False)
#features = VGGmodel.predict(xTrain)
#print(features.shape)

# we will add layers to this feature extraction part of DenseNet network
m = VGGmodel.output
# we start with a global average pooling
m = GlobalAveragePooling2D()(m)
# and add a fully-connected layer
m = Dense(1024, activation='relu')(m)
# finally, the softmax layer for predictions (we have nbClasses classes)
predictions = Dense(nbClasses, activation='softmax')(m)

# global network
model = Model(inputs=VGGmodel.input, outputs=predictions)
#short_summary(model)

# training
ourCallback = tensorflow.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001, patience=20, verbose=0, 
    mode='auto', baseline=None, restore_best_weights=False)

# training part I: training only the classification part (the end)
for layer in VGGmodel.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
#to prenvent it from too long a run
history=model.fit(
    xTrain, yTrain, epochs=100, batch_size=32, validation_split=0.4, 
    callbacks=[ourCallback],verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Chart')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.ylim([0, 1.1]) # Set the vertical axis range from 0 to 1
plt.grid() # Add a grid to the plot
plt.legend(['Train_accuracy', 'Val_accuracy'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Chart')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.ylim([0, 1.1]) # Set the vertical axis range from 0 to 1
plt.grid() # Add a grid to the plot
plt.legend(['Train_loss', 'Val_loss'], loc='upper left')
plt.show()

#Prediction evaluation of the model 
score = model.evaluate(xTest,yTest)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#Confusion Matrix and heatmap formulation
yPred = model.predict(xTest)
yPredV = yPred.argmax(axis=1)
yTestV = yTest.argmax(axis=1)
err = yTestV - yPredV
err
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTestV, yPredV)
print(cm)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your original confusion matrix is stored in the variable `cm` with shape (4, 4)

df_cm = pd.DataFrame(cm, index=[i for i in range(12)], columns=[i for i in range(12)])
plt.figure(figsize=(5, 4))
sn.heatmap(df_cm, annot=True)