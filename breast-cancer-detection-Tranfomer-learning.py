import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
import pandas as pd
import numpy as np
import os
from glob import glob
import random
import matplotlib.pylab as plt
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG16,VGG19,ResNet101V2,InceptionV3,DenseNet121,EfficientNetB7
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from matplotlib.image import imread
import cv2
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from plot_keras_history import show_history, plot_history



imagePatches = glob('C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/IDC_regular_ps50_idx5/**/*.png', recursive=True)
# for filename in imagePatches[0:10]:
#     print(filename)
class0 = [] 
class1 = [] 

for filename in imagePatches:
    if filename.endswith("class0.png"):
         class0.append(filename)
    else:
        class1.append(filename)
print(len(class0))
print(len(class1))
sampled_class0 = random.sample(class0, 78786)
sampled_class1 = random.sample(class1, 78786)
print(len(sampled_class0))
print(len(sampled_class1))
img_size = 75
def get_image_arrays(data, label):
    img_arrays = []
    for i in data:
        if i.endswith('.png'):
            img = cv2.imread(i ,cv2.IMREAD_COLOR)
            img_sized = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_LINEAR)
            img_arrays.append([img_sized, label])
    return img_arrays
class0_array = get_image_arrays(sampled_class0, 0)
class1_array = get_image_arrays(sampled_class1, 1)
test = cv2.imread('C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/IDC_regular_ps50_idx5/13689/1/13689_idx5_x801_y1501_class1.png' ,cv2.IMREAD_COLOR)
test.shape
combined_data = np.concatenate((class0_array, class1_array))
random.seed(41)
random.shuffle(combined_data)
X = []
y = []
for features,label in combined_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#------------------------------------------VGG16------------------------------
base_model = VGG16(weights='imagenet', include_top=False,input_shape=(img_size, img_size,3))
# freeze extraction layers
base_model.trainable = False
# add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(4096,activation="relu")(x)
x = Dense(4096,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2048,activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in model.layers:
    if layer.trainable==True:
        print(layer)
plot_model(model, to_file='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/VGG16_convnet.png', show_shapes=True,show_layer_names=True)
# Image(filename='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/convnet.png') 
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]          
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history=model.fit(X_train, y_train,validation_data=(X_test, y_test),verbose = 1,epochs = 30,callbacks=callbacks)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/Training_history VGG16.png",
             title="Training history VGG16")
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("VGG16 Confusion Matrix")
plt.savefig('C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/VGG16 predict confusion matrix.png') 
plt.show()
print(classification_report(Y_true, Y_pred_classes))


#------------------------------------------VGG19------------------------------
base_model = VGG19(weights='imagenet', include_top=False,input_shape=(img_size, img_size,3))
# freeze extraction layers
base_model.trainable = False
# add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(4096,activation="relu")(x)
x = Dense(4096,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2048,activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in model.layers:
    if layer.trainable==True:
        print(layer)
plot_model(model, to_file='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/VGG19_convnet.png', show_shapes=True,show_layer_names=True)
# Image(filename='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/convnet.png') 
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]          
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history=model.fit(X_train, y_train,validation_data=(X_test, y_test),verbose = 1,epochs = 30,callbacks=callbacks)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/Training_history VGG19.png",
             title="Training history VGG19")
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("VGG19 Confusion Matrix")
plt.savefig('C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/VGG19 predict confusion matrix.png') 
plt.show()
print(classification_report(Y_true, Y_pred_classes))


#------------------------------------------ResNet101V2------------------------------
base_model = ResNet101V2(weights='imagenet', include_top=False,input_shape=(img_size, img_size,3))
# freeze extraction layers
base_model.trainable = False
# add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(4096,activation="relu")(x)
x = Dense(4096,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2048,activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in model.layers:
    if layer.trainable==True:
        print(layer)
plot_model(model, to_file='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/ResNet101V2_convnet.png', show_shapes=True,show_layer_names=True)
# Image(filename='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/convnet.png') 
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]          
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history=model.fit(X_train, y_train,validation_data=(X_test, y_test),verbose = 1,epochs = 30,callbacks=callbacks)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/Training_history ResNet101V2.png",
             title="Training history ResNet101V2")
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("ResNet101V2 Confusion Matrix")
plt.savefig('C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/ResNet101V2 predict confusion matrix.png') 
plt.show()
print(classification_report(Y_true, Y_pred_classes))


#------------------------------------------InceptionV3------------------------------
base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(img_size, img_size,3))
# freeze extraction layers
base_model.trainable = False
# add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(4096,activation="relu")(x)
x = Dense(4096,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2048,activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in model.layers:
    if layer.trainable==True:
        print(layer)
plot_model(model, to_file='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/InceptionV3_convnet.png', show_shapes=True,show_layer_names=True)
# Image(filename='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/convnet.png') 
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]          
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history=model.fit(X_train, y_train,validation_data=(X_test, y_test),verbose = 1,epochs = 30,callbacks=callbacks)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/Training_history InceptionV3.png",
             title="Training history InceptionV3")
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("InceptionV3 Confusion Matrix")
plt.savefig('C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/InceptionV3 predict confusion matrix.png') 
plt.show()
print(classification_report(Y_true, Y_pred_classes))


#------------------------------------------DenseNet121------------------------------
base_model = DenseNet121(weights='imagenet', include_top=False,input_shape=(img_size, img_size,3))
# freeze extraction layers
base_model.trainable = False
# add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(4096,activation="relu")(x)
x = Dense(4096,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2048,activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in model.layers:
    if layer.trainable==True:
        print(layer)
plot_model(model, to_file='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/DenseNet121_convnet.png', show_shapes=True,show_layer_names=True)
# Image(filename='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/convnet.png') 
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]          
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history=model.fit(X_train, y_train,validation_data=(X_test, y_test),verbose = 1,epochs = 30,callbacks=callbacks)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/Training_history DenseNet121.png",
             title="Training history DenseNet121")
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("DenseNet121 Confusion Matrix")
plt.savefig('C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/DenseNet121 predict confusion matrix.png') 
plt.show()
print(classification_report(Y_true, Y_pred_classes))

#------------------------------------------EfficientNetB7------------------------------
base_model = EfficientNetB7(weights='imagenet', include_top=False,input_shape=(img_size, img_size,3))
# freeze extraction layers
base_model.trainable = False
# add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(4096,activation="relu")(x)
x = Dense(4096,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2048,activation="relu")(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in model.layers:
    if layer.trainable==True:
        print(layer)
plot_model(model, to_file='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/EfficientNetB7_convnet.png', show_shapes=True,show_layer_names=True)
# Image(filename='C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/convnet.png') 
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]          
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history=model.fit(X_train, y_train,validation_data=(X_test, y_test),verbose = 1,epochs = 30,callbacks=callbacks)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/Training_history EfficientNetB7.png",
             title="Training history EfficientNetB7")
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("EfficientNetB7 Confusion Matrix")
plt.savefig('C:/Users/GIGABYTE/Downloads/Breast Histopathology Images/img/EfficientNetB7 predict confusion matrix.png') 
plt.show()
print(classification_report(Y_true, Y_pred_classes))
