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
from helpers import adjust_brightness, weighted_BCE, focal_loss, alpha_focal_loss, soft_dice_loss, iou_loss, Metrics
plt.style.use("ggplot")

###########
# K.James 2019
#
# Trains a U-Net model using k-fold validation. 
# This code used for MSc thesis and paper in various configurations.
# For paper: this script used for WBCE and WBCE with brightness augmentation
#            use -l= "w_bce" and toggle -b between False and True 

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

def plot_sample(X, y, preds, binary_preds, ix=None):
    # For quick comparisons
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


#   Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--loss", type=str, default="w_bce")
ap.add_argument("-b", "--bright", type=bool, default=True)
ap.add_argument("-g", "--gamma", type=float, default=0)
args = vars(ap.parse_args())


#   Set some parameters
im_width = 512
im_height = 512
border = 5
path_train = 'unet_data_new\\'
path_test = 'unet_data_new\\test\\'
#------------------------------------------



#Train -------
X, y = get_data(path_train, load_masks=True)
kfold = KFold(n_splits=10, shuffle=True, random_state=42) #<----------------------------------------------------------------
fold = 0
overall_metrics = [[],[],[],[],[],[],[],[],[],[],[]]
names = ["precision","recall","F1","Dice","IOU","Accuracy","av_precision","av_recall","av_F1","av_Dice","av_IOU"]
#------------

LOSS = {
    "binary_crossentropy":"binary_crossentropy",
    "w_bce": weighted_BCE(w=[0.08,0.92]),       #these values set for this dataset's training set
    "focal": focal_loss(gamma=args["gamma"]),
    "alpha_focal":alpha_focal_loss(w=[0.08,0.92],gamma=args["gamma"]),
    "dice":soft_dice_loss,
    "iou":iou_loss
}

for train_indices, val_indices in kfold.split(X, y):
    fold+=1
    print("\n \n Fold {}".format(fold))
        
    lossfn = args["loss"]
    print(lossfn)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_valid = X[val_indices]
    y_valid = y[val_indices] 

    print(len(X_train),len(X_valid))

    input_img = Input((im_height, im_width, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(optimizer=sgd, loss=LOSS[lossfn], metrics=['accuracy'])

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
        ModelCheckpoint('kfold\\output\\model-hakea{}.h5'.format(fold), verbose=1, save_best_only=True, save_weights_only=True)
    ]
   
    results = model.fit_generator(generator(), steps_per_epoch=(len(X_train) // bs), epochs=2000, callbacks=callbacks,
                                validation_data=(X_valid, y_valid)) #<----------------------------------------------------------------

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('kfold\\output\\LearningCurve{}'.format(fold),bbox_inches='tight')

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="accuracy")
    plt.plot(results.history["val_accuracy"], label="val_accuracy")
    plt.plot( np.argmin(results.history["val_loss"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('kfold\\output\\LearningCurveAcc{}'.format(fold),bbox_inches='tight')

    #---------

    model.load_weights('kfold\\output\\model-hakea{}.h5'.format(fold))

    
    preds_val = model.predict(X_valid, verbose=1)

    
    kernel = np.ones((5,5),np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    binary = []

    #remove noise
    for i in range(0,len(preds_val_t)):
        im = preds_val_t[i]   
        
        im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)      
            
        binary.append(im)
      
           
    # Check if valid data looks all right
    '''num = len(X_valid)    
    for i in range(0,num):    
        plot_sample(X_valid, y_valid, preds_val, binary, ix=i) '''
    
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
