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
#K.James 2018
#
#Trains a U-Net model 
# Arguments:
# Images in paths: path_train = 'unet_data_new\\' and  path_test = 'unet_data_new\\test\\'
# Expects images to be in 'images' folder and corresponding masks to be in 'masks' folder within this directory
#
#References: 
# U-Net: O. Ronneberger, P. Fischer, and T.Brox, "U-Net: Convolutional netowrks for biomedical image segmentation," 
# in International Conference on Meidcal image computing and computer-assisted intervention. 
# Springer, 2015, pp.234-241
# Acknowledgements: This code is expanded from the tutorial at:  https://www.depends-on-the-definition.com/unet-keras-segmenting-images/ (has MIT license).

###########

#turn off GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Args for mode. Options: train, test, train final model for inference (inf)
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--state", type=str, default="train")
ap.add_argument("-w", "--weights", type=str, default="1,0.5")
ap.add_argument("-t", "--thick", type=int, default=1)
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

#------------------------------------------


def adjust_brightness(image):
    #cv2.imshow("i",image)
    #cv2.waitKey(0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2]=hsv[:,:,2]*random.randrange(30,100,1)/100 #changed from 3 10 to 30 100
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #cv2.imshow("i",out)
    #cv2.waitKey(0)
    return out

def Histogram(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])    
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)    
    return image

# Get and resize images and masks
def get_data(path, train=True):
    #ids = next(os.walk(path + "images"))[2]
    ids = os.listdir(path + "images") 
    if(train):
        idsmasks  = os.listdir(path + "masks")  
    l = len(ids)
    l=84
    #print(l)
    #exit()
    
    X = np.zeros((l, im_height, im_width, 3), dtype=np.float32) 
   
    if train:
        y = np.zeros((l, im_height, im_width, 1), dtype=np.float32)    
        
    print('Getting and resizing images ... ')
    for i in range(l):
        
        # Load images        
        image = cv2.imread(path + "images\\" + ids[i])
                
        #image = Histogram(np.array(img))       
        #CROP IMAGE to 1:1 aspect ratio        
        #x_img = image[:,500:3500]     
        #cv2.imwrite("data1\{}.jpg".format(i),x_img)
        #continue
        
        x_img = cv2.resize(image,(im_height, im_width))
        
             
        # Load masks
        if train:
            #mask = img_to_array(load_img(path + 'masks\\' + idsmasks[i], grayscale=True))    
            mask = cv2.imread(path + 'masks\\' + idsmasks[i],0)       
            
            #mask= mask[:,500:3500]     #Crop to same aspect ratio as image
            
            mask = cv2.resize(mask,(im_height, im_width))
            
            ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY) #thresh to fix any px that arent quite 1 or 0 after resize
           
            '''nzero = np.transpose(np.nonzero(mask))
           
            for j in nzero:
                if mask[j[0]][j[1]]/255 != 1:
                    print(mask[j[0]][j[1]]/255)'''        
            

            
            
            mask = mask.reshape((im_height, im_width,1))

        
        
        X[i] = x_img.squeeze() / 255
       
        if train:
            y[i] = mask / 255
    print('Done!')
    if train:
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

def get_unet(n_filters=16, dropout=0.5, batchnorm=True):
    
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

def get_unet2(input_img, n_filters=16, dropout=0.5, batchnorm=True):
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

def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))     
    
    #plot
    has_mask = y[ix].max() > 0
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix])
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Input with output overlay')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Mask')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Prediction')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Prediction binary')
    plt.savefig("output\{}.png".format(ix),bbox_inches='tight')
    plt.close()
# -------------------------------------------------
if(args["state"]=="train"):
    #Train -------
    X, y = get_data(path_train, train=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True) # Split train and valid
    print(len(X_train))
    
    w_train = generate_weight_maps(y_train,[U,L],t)
    w_train = np.expand_dims(w_train,3)     
elif(args["state"]=="test"):   
    X_valid, y_valid = get_data(path_test, train=True)
elif(args["state"]=="inf"):
    #Inference#------- Use all of dataset 1 to train final model for inference
    X_train, y_train = get_data(path_train, train=True)    
    
    w_train = generate_weight_maps(y_train,[U,L],t)
    w_train = np.expand_dims(w_train,3)
else:
    print("CMD -l options: train,test,inf (train for inference)")
    print("CMD -l options: train,test,inf (train for inference)")
    y_train = np.array([[[0.5,2,3,4],[2,5,8,2],[0.5,2,3,4],[2,5,8,2]],[[1,1,1,2],[2,4,6,7],[0.5,2,3,4],[2,5,8,2]]])
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
    exit()


#Verify weights all =1 gives same as BCE 

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
#-----------Comment out for testing

#Datagen


if(args["state"] == "train" or args["state"]=="inf"):

    model = get_unet( n_filters=16, dropout=0.05, batchnorm=True)
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
        results = model.fit_generator(generator(), steps_per_epoch=(len(X_train) // bs), epochs=150, callbacks=callbacks) 
        model.save_weights('model-hakea.h5')
        exit()
    else:    
        w_valid = generate_weight_maps(y_valid,[U,L],t) 
        #calculate weight map on the fly #KATHERINE - THIS IS REALLY IMPORTANT. SHOULD THIS BE WITH OR WITHOUT CW?
        #If it goes through the loss fn, it should be without. If not should add inverse class here.
        #have a look at all codes and see if this is inverse_class_weightmap or not.
        w_valid = np.expand_dims(w_train,3) 
        a =  [X_valid,w_valid]
        b =  [y_valid,w_valid]     
        results = model.fit_generator(generator(), steps_per_epoch=(len(X_train) // bs), epochs=150, callbacks=callbacks, validation_data=(a,b))

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
input_img = Input((im_height, im_width, 3), name='img')
inference = get_unet2(input_img, n_filters=16, dropout=0.05, batchnorm=True)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
inference.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy']) 
#note: bce is simply used to compile a the model now that it has the right shape
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