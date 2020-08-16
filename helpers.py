from keras import backend as K
import numpy as np
import tensorflow as tf
import cv2
import random



im_width = 512
im_height = 512



#------WEIGHT MAPS AND MAP LOSS----#

def cw_map_loss(groundtruth,predicted): #weight map based loss with class weighting
    y_true,weights = groundtruth[0],groundtruth[1]
    y_pred=predicted[0] #can only handle even number of images

    class_imbalance = [0.92,0.08]
    y_true_inv = 1 - y_true
    class_weights = y_true_inv*class_imbalance[1] + y_true*class_imbalance[0]


    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon()) 
    loss = -y_true * K.log(y_pred) - (1.0 -y_true) * K.log(1.0 - y_pred) 
    
    loss = loss * class_weights * weights
    return K.mean(loss)

def generate_weight_maps(y,w,t):
    #Use this function to pre-compute weight maps to downweight edges of objects when 
    #the annotation of the mask at the edge of objects has uncertainty

    #Parameters:
    #y is binary mask [0:1]
    #w is int array weightings: [Other,Edge] 
    #t is int thickness

    weight_maps = []
    for i in range(0,len(y)):
       
        mask = np.uint8(y[i].squeeze()*255) #remove extra dimension and scale up to regular pixel range       
        ret, mask = cv2.threshold(mask, 127, 255, 0)   #just in case
        
        canvas = np.ones(mask.shape)
        canvas = canvas * w[0] * 255    #set the majority of the weight matrix to reflect the non-edge part of the image        
        edge = w[1] *255 #edge weight     

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
        cv2.drawContours(canvas, contours, -1 , (edge, edge, edge),t)  
                                     
        canvas = canvas/255 # adjust to range [0:1]     
                    
        weight_maps.append(canvas)       
 
    return np.array(weight_maps)

def generate_class_weighted_maps(y,w,t):
    #Use this function to pre-compute weight maps to downweight edges of objects when 
    #the annotation of the mask at the edge of objects has uncertainty.
    #This version also accounts for class imbalance in the precomputed map. 

    #Parameters:
    #y is binary mask [0:1]
    #w is int array weightings: [Other,Target,Edge] 
    #t is int thickness

    weight_maps = []
    for i in range(0,len(y)):
       
        mask = np.uint8(y[i].squeeze()*255) #remove extra dimension and scale up to regular pixel range       
        ret, mask = cv2.threshold(mask, 127, 255, 0)   #just in case
        
        canvas = np.ones(mask.shape)
        canvas = canvas * w[0] * 255    #set the majority of the weight matrix to reflect the non-edge part of the image        
        
        target = w[1]*255 #target weight
        cv2.drawContours(canvas, contours, -1 , (target, target, target),-1)

        edge = w[2]*255 #edge weight     
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
        cv2.drawContours(canvas, contours, -1 , (edge, edge, edge),t)  
                                     
        canvas = canvas/255 # adjust to range [0:1]     
                    
        weight_maps.append(canvas)       
 
    return np.array(weight_maps)


#------USEFUL FUNCTIONS----#
def Histogram(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])    
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)    
    return image

def adjust_brightness(image):    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2]=hsv[:,:,2]*random.randrange(30,100,1)/100 
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  
    return out

def plot_sample(X, y, preds, binary_preds, ix=None):
    # For quick comparisons - requires folder 'output'
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


#------LOSS FUNCTIONS----#
def weighted_BCE(w):
    #BINARY CROSS ENTROPY WEIGHTED. EXPECTS W = [W0,W1] WHERE W0 IS BACKGROUND CLASS
 
    def loss(y_true,y_pred):   
        y_true_inv = 1 - y_true
        weights = y_true_inv*w[0] + y_true*w[1]              
            
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
      
        loss = -y_true * K.log(y_pred) - (1.0 -y_true) * K.log(1.0 - y_pred) 
       
        
        loss = loss * weights   #At this point it is pixel wise.       
                
        return K.mean(loss) 
    return loss 
   
def focal_loss(gamma): 
    #(https://arxiv.org/abs/1708.02002). 

    def loss(y_true,y_pred):         
              
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())   
        loss = -y_true * K.log(y_pred) * K.pow(1.-y_pred,gamma)    - (1.0 -y_true) * K.log(1.0 - y_pred)*K.pow(y_pred,gamma) # -log(pred) * y_TP - log(1-pred)*y_TN    
        return K.mean(loss) 
    return loss

def alpha_focal_loss(w,gamma):   
    #(https://arxiv.org/abs/1708.02002). 
    #WEIGHTED. EXPECTS W = [W0,W1] WHERE W0 IS BACKGROUND CLASS
   
    def loss(y_true,y_pred):
         
        y_true_inv = 1 - y_true
        weights = y_true_inv*w[0] + y_true*w[1] 
            
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())   
        loss = -y_true * K.log(y_pred) * K.pow(1.-y_pred,gamma) - (1.0 -y_true) * K.log(1.0 - y_pred)*K.pow(y_pred,gamma)  
        loss = loss*weights
        return K.mean(loss) 

    return loss

def soft_dice_loss(y_true,y_pred):
    return 1 -dice_coeff(y_true,y_pred)
    #https://www.jeremyjordan.me/semantic-segmentation/#advanced_unet

def dice_coeff(y_true,y_pred,epsilon=1e-6):   
    axes = tuple(range(1, len(y_pred.shape)))     
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes) 
    
    temp = (numerator*1.0+epsilon)/(denominator+epsilon)    
    return temp
    #return K.mean(temp) # average over classes and batch

def iou_loss(y_true,y_pred):
    return 1 - iou(y_true,y_pred)

def iou(y_true,y_pred,epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape))) 
   
    inter = K.sum(y_pred * y_true, axes)
    union = K.sum(y_true + y_pred - y_pred*y_true, axes)       
    temp = (inter*1.0+epsilon)/(union+epsilon)    
    #return K.mean(temp) # average over classes and batch
    return temp

#------EVALUATION----#

class Metrics:
    # class for calculation of all metrics for each image 
    # batchwise: calculate across batch
    # average: calc for each image, average across batch

    def __init__(self,y_true,y_pred):
       self.y_true = y_true
       self.y_pred = y_pred  
       #self.y_pred = np.array([0,0,0,1] )
       #self.y_true = np.array([1,1,1,1] )           
   
    

    def prop(self):
        axes = tuple(range(1, len(self.y_pred.shape))) 
        score_p = np.sum(self.y_pred * self.y_true, axes) 
        score_n = np.sum((1-self.y_pred) * (1-self.y_true), axes)
        score = score_p/(np.sum(self.y_true)) + score_n/(np.sum((1-self.y_true)))
        #print(score_p,np.sum(self.y_true),score_n,np.sum(1-self.y_true))
        return np.mean(score)

    def dice(self,batchwise=False):  
        
        if(batchwise):
            numerator = 2. * np.sum(self.y_pred * self.y_true)       
            denominator = np.sum(np.square(self.y_pred) + np.square(self.y_true))
            
        else:
            axes = tuple(range(1, len(self.y_pred.shape))) 
            #print(axes)
            numerator = 2. * np.sum(self.y_pred * self.y_true, axes)       
            denominator = np.sum(np.square(self.y_pred) + np.square(self.y_true), axes)

            
        
        temp = numerator*1.0/denominator
        #temp = numerator*1.0/(denominator+1e-6)   
        
        temp = temp[~np.isnan(temp)] 
        return np.mean(temp) # average over classes and batch
    
    def iou(self,batchwise=False):  
        epsilon=1e-6  
        #K.sum(y_true + y_pred - y_pred*y_true 
        if(batchwise):
            inter = np.sum(self.y_pred * self.y_true)
            union = np.sum(self.y_true + self.y_pred - (self.y_pred*self.y_true))  
            #return  inter / (union + epsilon) 
        else:
            axes = tuple(range(1, len(self.y_pred.shape))) 
            inter = np.sum(self.y_pred * self.y_true, axes)
            union = np.sum(self.y_true + self.y_pred - (self.y_pred*self.y_true), axes)       
        temp = inter*1.0/union
        temp = temp[~np.isnan(temp)] 
        return np.mean(temp) # average over classes and batch
    
    def accuracy(self,batchwise=False):    
                             
        t0 = np.equal(self.y_true,np.round(self.y_pred))                   
        t1 = np.mean(t0,axis=0)         
                                   
        return np.mean(t1)            
            
    def precison(self,batchwise=False):
        epsilon=1e-6 
        if(batchwise):
            #batchwise precision                        
            true_positives = np.sum(self.y_pred * self.y_true)
            all_positives = np.sum(self.y_pred)
            
            #return true_positives/(all_positives+epsilon)
        else:
            #average        
            axes = tuple(range(1, len(self.y_pred.shape))) #this lets us do each image separately
            true_positives = np.sum(self.y_pred * self.y_true,axes)
            #print("****DEBUG****",np.sum(true_positives))
            all_positives = np.sum(self.y_pred,axes)
        temp = true_positives*1.0/(all_positives)
        temp = temp[~np.isnan(temp)]        
        return np.mean(temp)
                   
    def recall(self,batchwise=False):
        epsilon=1e-6 
        if(batchwise):
            #batchwise recall                         
            true_positives = np.sum(self.y_pred * self.y_true)
            actual_positives = np.sum(self.y_true)            
            #return true_positives/(actual_positives+epsilon)
        
        else:
            #average        
            axes = tuple(range(1, len(self.y_pred.shape))) #this lets us do each image separately
            true_positives = np.sum(self.y_pred * self.y_true,axes)
            actual_positives = np.sum(self.y_true,axes)       
        temp = true_positives*1.0/(actual_positives)       
        temp = temp[~np.isnan(temp)]        
        return np.mean(temp)

    def f1_score(self,batchwise=False):
        epsilon=1e-6  
        if(batchwise):
            #batchwise f1            
            true_positives = np.sum(self.y_pred * self.y_true)            
            all_positives = np.sum(self.y_pred)
            precision=  true_positives/(all_positives+epsilon)            

            actual_positives = np.sum(self.y_true)
            recall =  true_positives/(actual_positives+epsilon)           

            #return 2. * precision* recall/((precision+recall)+epsilon)
        else:
            #average
            axes = tuple(range(1, len(self.y_pred.shape))) #this lets us do each image separately           
            true_positives = np.sum(self.y_pred * self.y_true,axes)            
            all_positives = np.sum(self.y_pred,axes)
            precision=  true_positives/(all_positives)            

            actual_positives = np.sum(self.y_true,axes)
            recall =  true_positives/(actual_positives)           
        
        temp = 2. * precision* recall/(precision+recall)
        temp = temp[~np.isnan(temp)]   
        return np.mean(temp)    
    
    def specificity(self,batchwise=False):
        epsilon=1e-6 
        y_pred = 1 - self.y_pred    
        y_true = 1 - self.y_true 
        if(batchwise):
            #batchwise recall                  
            true_negatives = np.sum((y_pred) * y_true)
            actual_negatives = np.sum(y_true)            
            #return true_positives/(actual_positives+epsilon)
        
        else:
            #average        
            axes = tuple(range(1, len(self.y_pred.shape))) #this lets us do each image separately
            true_negatives = np.sum(y_pred * y_true,axes)
            actual_negatives = np.sum(y_true,axes)       
        temp = true_negatives*1.0/(actual_negatives)       
        temp = temp[~np.isnan(temp)]        
        return np.mean(temp)
    
    

        

