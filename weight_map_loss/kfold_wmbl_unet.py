import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Dropout, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Reshape
from keras.layers.merge import add, concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold, train_test_split

#from helpers import *
from helpers import adjust_brightness, Metrics, cw_map_loss, generate_class_weighted_maps, generate_weight_maps
plt.style.use("ggplot")

###########
# K.James 2019
#
# Trains a U-Net model using k-fold validation. 
# This code used for MSc thesis and paper in various configurations.

# Arguments: 
# -w : weight map weights
# -t : weight map edge thickness
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

    model = Model(inputs=[input_img], outputs=[outputs])
  
    return model

#   Arguments
ap = argparse.ArgumentParser()
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
#F=weights[1]
#L=weights[2]
#------------------------------------------

#Train -------
X, y = get_data(path_train, load_masks=True)
kfold = KFold(n_splits=10, shuffle=True, random_state=42) #<----------------------------------------------------------------
fold = 0
overall_metrics = [[],[],[],[],[],[],[],[],[],[],[]]
names = ["precision","recall","F1","Dice","IOU","Accuracy","av_precision","av_recall","av_F1","av_Dice","av_IOU"]
#------------

for train_indices, val_indices in kfold.split(X, y):
    fold+=1
    print("\n \n Fold {}".format(fold))  

    
    print(train_indices.shape,val_indices.shape)
    if(len(train_indices)%2 !=0):
        train_indices = train_indices[:-1]
    if(len(val_indices)%2 !=0):
        val_indices = val_indices[:-1]
    print(train_indices.shape,val_indices.shape)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_valid = X[val_indices]
    y_valid = y[val_indices] 

    w_train = generate_weight_maps(y_train,[U,L],t)    
    w_train = np.expand_dims(w_train,3)

    #w_valid = generate_class_weighted_maps(y_valid,[U,F,L],t) 
    w_valid = generate_weight_maps(y_valid,[U,L],t) 
    w_valid = np.expand_dims(w_valid,3)

    print(len(X_train),len(X_valid))
   
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
        ModelCheckpoint('kfold\\output\\model-hakea{}.h5'.format(fold), verbose=1, save_best_only=True, save_weights_only=True)
    ]
   
    a =  [X_valid,w_valid]
    b =  [y_valid,w_valid]     
    results = model.fit_generator(generator(), steps_per_epoch=(len(X_train) // bs), epochs=2000, callbacks=callbacks, validation_data=(a,b))    
    
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('kfold\\output\\LearningCurve{}'.format(fold),bbox_inches='tight')

    

    #---------
    input_img = Input((im_height, im_width, 3), name='img')
    inference = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
    inference.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])

    inference.load_weights('kfold\\output\\model-hakea{}.h5'.format(fold))
    
    preds_val = inference.predict(X_valid, verbose=1)
    
    kernel = np.ones((5,5),np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    binary = []

    #remove noise
    for i in range(0,len(preds_val_t)):
        im = preds_val_t[i]   
        
        im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)      
            
        binary.append(im)
   
           
    
    print('calculating metrics')
    y_valid = y_valid.astype('uint8')
    binary = np.array(binary)
    binary = binary.astype('uint8')
    
    m = Metrics(y_valid.squeeze(),binary) 
    overall_metrics[0].append(m.precison(True))
    overall_metrics[1].append(m.recall(True))
    overall_metrics[2].append(m.f1_score(True))
    overall_metrics[3].append(m.dice(True))
    overall_metrics[4].append(m.iou(True))
    overall_metrics[5].append(m.accuracy(True))
    
    overall_metrics[6].append(m.precison(False))
    overall_metrics[7].append(m.recall(False))
    overall_metrics[8].append(m.f1_score(False))
    overall_metrics[9].append(m.dice(False))
    overall_metrics[10].append(m.iou(False))    
    
    f = open("kfold\\output\\{}.txt".format(fold),"w+")
    f.write("           Batchwise,         Average\n")
    f.write("Precision {:.2f} {:.2f}\n".format(m.precison(True),m.precison(False)))
    f.write("Recall    {:.2f} {:.2f}\n".format(m.recall(True),m.recall(False)))
    f.write("F1        {:.2f} {:.2f}\n".format(m.f1_score(True),m.f1_score(False)))
    f.write("Dice      {:.2f} {:.2f}\n".format(m.dice(True),m.dice(False)))
    f.write("IOU       {:.2f} {:.2f}\n".format(m.iou(True),m.iou(False)))
    f.write("Accuracy  {:.2f}\n".format(m.accuracy()))
    f.close()
    
    
f = open("kfold\\output\\cumulative_results.txt","w+")

print('-------------------------')
print('-------------------------')
for m in range(len(overall_metrics)):  
        n = names[m]       
        mean = np.array(overall_metrics[m])            
        overall_metrics[m].append(np.mean(mean)) #add so we can take mean across the folds
        stdev = np.std(overall_metrics[m])
        print("{}: {:.2f} +- {:.2f}".format(n,np.mean(mean),stdev))
        f.write("{}: {:.2f} +- {:.2f}\n".format(n,np.mean(mean),stdev))

print('-------------------------')
print('-------------------------')
f.close()
