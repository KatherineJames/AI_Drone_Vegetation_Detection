import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.pooling import GlobalMaxPool2D, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator 

from sklearn.model_selection import train_test_split

from helpers import *
plt.style.use("ggplot")

###########
# K.James 2019
#
# Trains a U-Net model using for inference 
# This code used for MSc thesis and paper in various configurations.
# For paper: this script used for WBCE and WBCE with brightness augmentation
#            use -l= "w_bce" and toggle -b between False and True 
# Use -s = "inf" to train inference model and then -s="test" to evaluate against unseen test set

# Arguments: 
# -l : loss function
# -b : toggle brightness augmentation
# -g : gamma for focal loss
#
# Images in paths: path_train = 'unet_data_new\\' and  path_test = 'unet_data_new\\test\\'
# Expects images to be in 'images' folder and corresponding masks to be in 'masks' folder within each of these paths
# Please create folder structure kfold\output for the outputs.
#
# References: 
# U-Net: O. Ronneberger, P. Fischer, and T.Brox, "U-Net: Convolutional netowrks for biomedical image segmentation," 
# in International Conference on Meidcal image computing and computer-assisted intervention. 
# Springer, 2015, pp.234-241
#
# Acknowledgements: U-Net implementation based on:  https://www.depends-on-the-definition.com/unet-keras-segmenting-images/ (has MIT license)

###########

#Args for mode. Options: train, test, train final model for inference (inf)
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--state", type=str, default="inf")
ap.add_argument("-b", "--bright", type=bool, default=True)
ap.add_argument("-l", "--loss", type=str, default="w_bce")
ap.add_argument("-g", "--gamma", type=float, default=0)
ap.add_argument("-m","--model",type=str,default="model-hakea")
args = vars(ap.parse_args())


# Set some parameters
im_width = 512
im_height = 512
border = 5
path_train = 'unet_data_new\\'
path_test = 'unet_data_new\\test\\'

#------------------------------------------

# Get and resize images and masks
def get_data(path, load_masks=True):  
    ids = os.listdir(path + "images") 
    idsmasks  = os.listdir(path + "masks")  
    l = len(ids)   
    X = np.zeros((l, im_height, im_width, 3), dtype=np.float32)
   
    if load_masks:
        y = np.zeros((l, im_height, im_width, 1), dtype=np.float32)        
    print('Getting and resizing images ... ')
    for i in range(l):  
        
        image = cv2.imread(path + "images\\" + ids[i])      
        x_img = cv2.resize(image,(im_height, im_width))
                    
        
        if load_masks:               
            mask = cv2.imread(path + 'masks\\' + idsmasks[i],0)
            mask = cv2.resize(mask,(im_height, im_width))
            ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY) #thresh to fix any px that arent quite 1 or 0 after resize
            mask = mask.reshape((im_height, im_width,1))

        X[i] = x_img.squeeze() / 255
       
        if load_masks:
            y[i] = mask / 255
    print('Done!')
    if load_masks:
        return X, y
    else:
        return X

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1) #stride defaults to poolsize, ie, 2.
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path   
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)    

    model = Model(inputs=[input_img], outputs=[outputs])    
    return model
# -------------------------------------------------

if(args["state"]=="train"):   
    X, y = get_data(path_train, load_masks=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True) # Split train and valid             
elif(args["state"]=="test"):   
    X_valid,y_valid = get_data(path_test, load_masks=True)      
elif(args["state"]=="inf"):
    #Inference#------- Use all of dataset 1 to train final model for inference
    X_train, y_train = get_data(path_train, load_masks=True)   
else:
    print("CMD -l options: train,test,inf (train for inference)")
    exit()

LOSS = {
    "binary_crossentropy":"binary_crossentropy",
    "w_bce": weighted_BCE(w=[0.08,0.92]),
    "focal": focal_loss(gamma=args["gamma"]),
    "alpha_focal":alpha_focal_loss(w=[0.08,0.92],gamma=args["gamma"]),
    "dice":soft_dice_loss,
    "iou":iou_loss
}

input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
model.compile(optimizer=sgd, loss=LOSS[args["loss"]], metrics=['accuracy']) #run in test mode, then compare to BCE and make decision

if(args["state"] == "train" or args["state"]=="inf"):
    if(args["bright"]):
        print('using bright data_gen')
        data_gen_args_image = dict(horizontal_flip=True,
                            vertical_flip=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            fill_mode='nearest',
                            preprocessing_function=adjust_brightness)    
        
    else:
        data_gen_args_image = dict(horizontal_flip=True,
                            vertical_flip=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            fill_mode='nearest')

    data_gen_args_mask = dict(horizontal_flip=True,
                        vertical_flip=True,
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        zoom_range=0.2,
                        fill_mode='nearest'
                        )
    
    image_datagen = ImageDataGenerator(**data_gen_args_image)
    mask_datagen = ImageDataGenerator(**data_gen_args_mask)

    seed = 2019 #same seed for both as we have to distort mask in the same way
    bs = 2

    
    def generator():
        image_generator = image_datagen.flow(X_train, seed=seed, batch_size=bs, shuffle=True)
        mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)
        while True:
            X1i = image_generator.next()                            
            yi = mask_generator.next()    

            # ---threshold arrays---keep them binary not interpolated           
            yi[yi < 0.5] = 0
            yi[yi >= 0.5] = 1               
                       
            yield (X1i,yi)

    

    callbacks = [
        EarlyStopping(patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-hakea.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    

    if(args["state"]=="inf"):
        results = model.fit_generator(generator(), steps_per_epoch=(len(X_train) // bs), epochs=150) #number of epochs chosen from k-fold results 
        model.save_weights('model-hakea.h5')
        exit()
    else:
        results = model.fit_generator(generator(), steps_per_epoch=(len(X_train) // bs), epochs=2000, callbacks=callbacks, validation_data=(X_valid, y_valid))

if(args["state"] == "train"):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('LearningCurve',bbox_inches='tight')

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="accuracy")
    plt.plot(results.history["val_accuracy"], label="val_accuracy")
    plt.plot( np.argmin(results.history["val_loss"]), np.max(results.history["val_acc"]), marker="x", color="r", label="best model")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('LearningCurveAcc',bbox_inches='tight')


#---------
model.load_weights("{}.h5".format(args["model"]))
model.save('model.h5')

#Info for Android Application
#print([node.op.name for node in model.inputs])
#print(model.summary())
#exit()

preds_val = model.predict(X_valid, verbose=1) #predict

kernel = np.ones((5,5),np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8) #threshold predictions
binary = []

#remove noise
for i in range(0,len(preds_val_t)):
    im = preds_val_t[i]       
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)   
    binary.append(im)
    
num = len(X_valid)
binary = np.array(binary)

print("METRICS")

m = Metrics(y_valid.squeeze(),binary.squeeze())   

print("           Batchwise,         Average")
print("Precision {:.2f} {:.2f}".format(m.precison(True),m.precison(False)))
print("Recall    {:.2f} {:.2f}".format(m.recall(True),m.recall(False)))
print("F1        {:.2f} {:.2f}".format(m.f1_score(True),m.f1_score(False)))
print("Dice      {:.2f} {:.2f}".format(m.dice(True),m.dice(False)))
print("IOU       {:.2f} {:.2f}".format(m.iou(True),m.iou(False)))
print("Accuracy  {:.2f} ".format(m.accuracy(True)))

# Save images for later perusal
# requires 'output' folder with subfolders images preds binary and masks
for i in range(0,num):   
    im = cv2.cvtColor(X_valid[i], cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(im)    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("output\\images\\{}.png".format(i),bbox_inches='tight')
    plt.close() 
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))    
    ax.imshow(preds_val[i].squeeze(), vmin=0, vmax=1)   
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("output\\preds\\{}.png".format(i),bbox_inches='tight')
    plt.close()   
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(binary[i].squeeze(), vmin=0, vmax=1)   
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("output\\binary\\{}.png".format(i),bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(y_valid[i].squeeze(), vmin=0, vmax=1)   
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("output\\masks\\{}.png".format(i),bbox_inches='tight')
    plt.close()

