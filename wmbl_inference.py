import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Input, Lambda, multiply)
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
# -s : state (train/inf/test)
# -w : weight map weights
# -t : weight map edge thickness
#
# Images in paths: path_train = 'unet_data_new\\' and  path_test = 'unet_data_new\\test\\'
# Expects images to be in 'images' folder and corresponding masks to be in 'masks' folder within each of these paths
# Please create folder structure \output for the outputs.
#
# References: 
# U-Net: O. Ronneberger, P. Fischer, and T.Brox, "U-Net: Convolutional netowrks for biomedical image segmentation," 
# in International Conference on Meidcal image computing and computer-assisted intervention. 
# Springer, 2015, pp.234-241
#
# Acknowledgements: U-Net implementation based on:  https://www.depends-on-the-definition.com/unet-keras-segmenting-images/ (has MIT license)

###########

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--state", type=str, default="inf")
ap.add_argument("-w", "--weights", type=str, default="2,0.5")
ap.add_argument("-t", "--thick", type=int, default=5)
args = vars(ap.parse_args())

weights =list(map(float,args["weights"].split(',')))

# Set some parameters
im_width = 512
im_height = 512
border = 5
path_train = 'unet_data_new\\'
path_test = 'unet_data_new\\test\\'
U = weights[0] #background
L = weights[1] #edge
t = args["thick"]
print(U,L,t)


def verify():
    # Verify weights all =1 gives same as BCE 
    '''
    from keras.objectives import binary_crossentropy
    loss = K.mean(binary_crossentropy(K.variable(y_train),K.variable(y_train*0.6))).eval(session=K.get_session())
    print(y_train.shape)
    y_true =   np.stack((y_train,w_train),axis=0)
    y_pred =   np.stack((y_train*0.6,w_train),axis=0)
    loss_map=ronneberg(K.variable(y_true),K.variable(y_pred)).eval(session=K.get_session())
    print('SHAPE',y_pred.shape)
    #np.testing.assert_almost_equal(loss_weighted,loss)
    #print('OK test1')
    print("BCE",loss)
    print("weightmap",loss_map)#,loss)
    exit()''' 
    '''
    MODEL INPUT [(None, 512, 512, 3), (None, 512, 512, 1)]
    MODEL OUTPUT [(None, 512, 512, 1), (None, 512, 512, 1)]
    GTRUTH (2, 10, 512, 512, 1)
    PRED (2, 10, 512, 512, 1)
    Length 10: 
    BCE 0.039709575
    weightmap 0.039709594'''
#------------------------------------------
# Get and resize images and masks
def get_data(path, load_masks=True):  
    ids = os.listdir(path + "images") 
    idsmasks  = os.listdir(path + "masks")  
    l = len(ids)   
    if(l%2 !=0):
        l=l-1
    l=8
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

def get_weightmap_unet(n_filters=16, dropout=0.5, batchnorm=True):
    
    input_img = Input((im_height, im_width, 3), name='img')
    weight_map_ip = Input((im_height, im_width,1),name='map')   
    
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
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid',name="output") (c9)

    model = Model(inputs=[input_img,weight_map_ip], outputs=[outputs,weight_map_ip]) #pass weight map straight from input to output
    #print(model.summary())
    print('MODEL INPUT',model.input_shape)
    print("MODEL OUTPUT",model.output_shape)   
    
    return model

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

    #weighted stuff
    #Add a few non trainable layers to mimic the computation of the crossentropy
    # loss, so that the actual loss function just has to peform the
    # aggregation.
    '''normalize_activation = Lambda(lambda x: x / tf.reduce_sum(x, len(x.get_shape()) - 1, True))(outputs)
    clip_activation = Lambda(lambda x: tf.clip_by_value(x, _epsilon, 1. - _epsilon))(normalize_activation)
    log_activation = Lambda(lambda x: K.log(x))(clip_activation)

    from keras import layers as L
    # Add a new input to serve as the source for the weight maps
    weight_map_ip = Input(shape=(im_height, im_width))
    weighted_softmax =L.multiply([log_activation, weight_map_ip])'''
    

    model = Model(inputs=[input_img], outputs=[outputs])
    #model = Model(inputs=[input_img], outputs=[weighted_softmax])
    return model

#------------------------------------------------
if(args["state"]=="train"):
    #Train -------
    print("TRAIN")
    X, y = get_data(path_train, load_masks=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True) # Split train and valid
    print(len(X_train))

    print("SHAPE", y_train.shape,y_valid.shape)
    
    w_train = generate_weight_maps(y_train,[U,L],t)    
    w_train = np.expand_dims(w_train,3)  

    #w_valid = generate_class_weighted_maps(y_valid,[U,F,L],t) 
    w_valid = generate_weight_maps(y_valid,[U,L],t) 
    w_valid = np.expand_dims(w_valid,3) 
   
     
elif(args["state"]=="test"):   
    print("TEST")
    X_valid, y_valid = get_data(path_test, load_masks=True)
elif(args["state"]=="inf"):
    #Inference#------- Use all of dataset 1 to train final model for inference
    print("INF")
    X_train, y_train = get_data(path_train, load_masks=True)    
    
    w_train = generate_weight_maps(y_train,[U,L],t)
    w_train = np.expand_dims(w_train,3)
else:
    print("CMD -l options: train,test,inf (train for inference)")    
    '''y_train = np.array([[[0.5,2,3,4],[2,5,8,2],[0.5,2,3,4],[2,5,8,2]],[[1,1,1,2],[2,4,6,7],[0.5,2,3,4],[2,5,8,2]]])
    w_train = np.array([[[1.0,1,1,1],[1,1,1,1],[1.0,1,1,1],[1,1,1,1]],[[1,1,1,1],[1,1,1,1],[1.0,1,1,1],[1,1,1,1]]])
    print(y_train.shape)
    
    from keras.objectives import binary_crossentropy
    loss = K.mean(binary_crossentropy(K.variable(y_train),K.variable(y_train*0.6))).eval(session=K.get_session())
    wbce = weighted_BCE([0.08,0.92])(K.variable(y_train),K.variable(y_train*0.6)).eval(session=K.get_session())

    y_true =   np.stack((y_train,w_train),axis=0)
    y_pred =   np.stack((y_train*0.6,w_train),axis=0)
    loss_map_bal=weighted_ronneberg(K.variable(y_true),K.variable(y_pred)).eval(session=K.get_session())
    loss_map=ronneberg(K.variable(y_true),K.variable(y_pred)).eval(session=K.get_session())

    print('SHAPE',y_pred.shape)
    #np.testing.assert_almost_equal(loss_weighted,loss)
    #print('OK test1')
    print("BCE",loss)
    print("WBCE",wbce)
    print("weightmap",loss_map)
    print("weightmap class balanced",loss_map_bal)#,loss)
    exit()'''


if(args["state"] == "train" or args["state"]=="inf"):

    model = get_weightmap_unet( n_filters=16, dropout=0.05, batchnorm=True)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(optimizer=sgd, loss=cw_map_loss) 

    
    data_gen_args_image = dict(horizontal_flip=True,
                        vertical_flip=True,
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        zoom_range=0.2,
                        fill_mode='nearest',
                        preprocessing_function=adjust_brightness)

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

      
    #yields [X1i, X2i],yi
    def generator():
        image_generator = image_datagen.flow(X_train, seed=seed, batch_size=bs, shuffle=True)
        mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)
        weight_generator = mask_datagen.flow(w_train, seed=seed, batch_size=bs, shuffle=True)
        while True:
            X1i = image_generator.next()
            X2i = weight_generator.next()                      
            yi = mask_generator.next()    

            # ---threshold arrays---keep them binary not interpolated

            #Weights
            M = (U-L)/2. + L
            X2i[X2i < M] = L
            X2i[X2i >= M] = U
            
            #Masks            
            yi[yi < 0.5] = 0
            yi[yi >= 0.5] = 1              
            
            yield ([X1i, X2i],[yi,X2i])
   
    callbacks = [
        EarlyStopping(patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-hakea.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    if(args["state"]=="inf"):           
        print("STEPS PER EPOCH: ",len(X_train) // bs) #how many batches we have       
        results = model.fit_generator(generator(), steps_per_epoch=(len(X_train) // bs), epochs=2, callbacks=callbacks) #set epochs=150
        model.save_weights('model-hakea.h5')
        exit()
    else:                   
        a =  [X_valid,w_valid]
        b =  [y_valid,w_valid]     
        results = model.fit_generator(generator(), steps_per_epoch=(len(X_train) // bs), epochs=2, callbacks=callbacks, validation_data=(a,b))

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

  

#---------
# Evaluate model on unseen test set
#---------
print("Evaluate on unseen test set")
input_img = Input((im_height, im_width, 3), name='img')
inference = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
inference.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy']) 
#note: bce is simply used to compile the model, now that it has the right shape
#ie. remove layers involved in loss function

inference.load_weights('model-hakea.h5')

#Info for Android Application
#print([node.op.name for node in model.inputs])
#print(inference.summary())

preds_val = inference.predict(X_valid, verbose=1) #predict

kernel = np.ones((5,5),np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8) #threshold predictions
binary = []

#remove noise
for i in range(0,len(preds_val_t)):
    im = preds_val_t[i]  
    #im = cv2.GaussianBlur(im, (3,3), 0)        
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    #im = cv2.GaussianBlur(im, (3,3), 0)      
    #im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    binary.append(im)
    
num = len(X_valid)
binary = np.array(binary)

print("METRICS")

m = Metrics(y_valid.squeeze(),binary.squeeze())   

print("           Batchwise,         Average")
print("Precision ",m.precison(True),m.precison(False))
print("Recall    ",m.recall(True),m.recall(False))
print("F1        ",m.f1_score(True),m.f1_score(False))
print("Dice      ",m.dice(True),m.dice(False))
print("IOU       ",m.iou(True),m.iou(False))
print("Accuracy  ",m.accuracy(True))


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