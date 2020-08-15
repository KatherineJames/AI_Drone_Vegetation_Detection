from keras import backend as K
import numpy as np
import tensorflow as tf
import cv2


im_width = 512
im_height = 512

BCEweights = []

#helper functions needed in both unet.py and kfold_unet.py
def ronneberg(y_true,y_pred):
    return -tf.reduce_sum(y_true*y_pred,len(y_pred.get_shape())-1)

def weighted_BCE(w):
    #BINARY CROSS ENTROPY WEIGHTED. EXPECTS W = [W0,W1] WHERE W0 IS BACKGROUND CLASS
    
    #weights = K.variable(weights)     
    #weights = K.expand_dims(weights,3)    
    def loss(y_true,y_pred):   
        y_true_inv = 1 - y_true
        weights = y_true_inv*w[0] + y_true*w[1]              
            
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
      
        loss = -y_true * K.log(y_pred) - (1.0 -y_true) * K.log(1.0 - y_pred) #8,512,512,1 (BCE)
        #loss = soft_dice_loss(y_true,y_pred)
        
        loss = loss * weights   #print shape, then determine if we need to sum or not. At this point it is pixel wise.
        
                
        return K.mean(loss) 
    return loss 

def weighted_BCE_MAP(weights):
    
    weights = K.variable(weights)     
    #weights = K.expand_dims(weights,3)
    
    def loss(y_true,y_pred):   
            
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
      
        loss = -y_true * K.log(y_pred) - (1.0 -y_true) * K.log(1.0 - y_pred) #8,512,512,1
        #loss = soft_dice_loss(y_true,y_pred)

        print(loss.shape,weights.shape)
        loss = loss * weights   
        
                
        return K.mean(loss)
    return loss 


def focal_loss(gamma): 

    def loss(y_true,y_pred):  #(https://arxiv.org/abs/1708.02002).        
              
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())   
        loss = -y_true * K.log(y_pred) * K.pow(1.-y_pred,gamma)    - (1.0 -y_true) * K.log(1.0 - y_pred)*K.pow(y_pred,gamma) # -log(pred) * y_TP - log(1-pred)*y_TN    
        return K.mean(loss) 
    return loss

 
def alpha_focal_loss(w,gamma):   #(https://arxiv.org/abs/1708.02002). 
    #WEIGHTED. EXPECTS W = [W0,W1] WHERE W0 IS BACKGROUND CLASS
   
    def loss(y_true,y_pred):
        #gamma=2      
        y_true_inv = 1 - y_true
        weights = y_true_inv*w[0] + y_true*w[1] 
            
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())   
        loss = -y_true * K.log(y_pred) * K.pow(1.-y_pred,gamma)    - (1.0 -y_true) * K.log(1.0 - y_pred)*K.pow(y_pred,gamma) # -log(pred) * y_TP - log(1-pred)*y_TN    
        loss = loss*weights
        return K.mean(loss) 

    return loss

def soft_dice_loss(y_true,y_pred):
    return 1 -dice_coeff(y_true,y_pred)
    #https://www.jeremyjordan.me/semantic-segmentation/#advanced_unet

def iou_loss(y_true,y_pred):
    return 1 - iou(y_true,y_pred)

def dice_coeff(y_true,y_pred,epsilon=1e-6):
    '''y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean(2. * intersection + epsilon / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon))'''
    axes = tuple(range(1, len(y_pred.shape)))     
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes) 
    
    temp = (numerator*1.0+epsilon)/(denominator+epsilon)    
    return temp
    #return K.mean(temp) # average over classes and batch


def iou(y_true,y_pred,epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape))) 
    #print(axes)
    inter = K.sum(y_pred * y_true, axes)
    union = K.sum(y_true + y_pred - y_pred*y_true, axes)       
    temp = (inter*1.0+epsilon)/(union+epsilon)    
    #return K.mean(temp) # average over classes and batch
    return temp

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    # From: https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def precision(y_true,y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    return K.mean(p)


def class_weight_map(y,w,t):
    #w is [Other,Edge weights] int array
    #t is int thickness

    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #ret, mask = cv2.threshold(mask, 127, 255, 0)
    #cv2.imshow('mask',mask)
    #cv2.waitKey(0)

    weights = []
    for i in range(0,len(y)):
        mask = np.uint8(y[i].reshape((im_height, im_width))*255)
        
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        canvas = np.ones(mask.shape)   

        canvas = canvas * w[0] * 255    #set the majority of the weight matrix to reflect the non-edge part of the image
        
        edge = w[1] *255 #edge weight
        #edge = 0
        
        cv2.drawContours(canvas, contours, -1 , (edge, edge, edge), -3) #-ve thickness means use a filled contour            
        '''cv2.imshow('i',canvas)    
        cv2.waitKey(0)    
        print(canvas) 
        exit()  '''
        #canvas = ((1-canvas/255)+1)/2  
        canvas = canvas/255
        
        #print(canvas)

        #canvas = np.ones(mask.shape)   #for verifying BCE=WBCE if weights=1
             
        weights.append(canvas)
       
    
    return np.array(weights)

def weight_map(y,w,t):
    #w is [Other,Edge weights] int array
    #t is int thickness

    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #ret, mask = cv2.threshold(mask, 127, 255, 0)
    #cv2.imshow('mask',mask)
    #cv2.waitKey(0)

    weights = []
    for i in range(0,len(y)):
        mask = np.uint8(y[i].reshape((im_height, im_width))*255)
        
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        canvas = np.ones(mask.shape)   

        canvas = canvas * w[0] * 255    #set the majority of the weight matrix to reflect the non-edge part of the image
        
        edge = w[1] *255 #edge weight
        
        cv2.drawContours(canvas, contours, -1 , (edge, edge, edge),5) #-ve thickness means use a filled contour                       
        #canvas = ((1-canvas/255)+1)/2  
        
        canvas = canvas/255
        
        #print(canvas)

        #canvas = np.ones(mask.shape)   #for verifying BCE=WBCE if weights=1
             
        weights.append(canvas)
       
    
    return np.array(weights)

   
'''def f1(y_true, y_pred):
    epsilon=1e-6
   
    #y_pred = np.round(y_pred)
    #print(y_true.shape)
    tp = np.sum(y_true*y_pred, axis=(1,2))
    
    tn = np.sum((1-y_true)*(1-y_pred), axis=(1,2))
    fp = np.sum((1-y_true)*y_pred, axis=(1,2))
    fn = np.sum(y_true*(1-y_pred), axis=(1,2))

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2*p*r / (p+r+epsilon)
    #print(f1.shape)
    #f1 =np.where(np.is_nan(f1), np.zeros_like(f1), f1)
    return f1

def iou(y_true,y_pred,epsilon=1e-6):
    axes = (1,2)
    inter = np.sum(y_pred * y_true,axes)
    #print(inter.shape)
    union = np.sum(y_true + y_pred - y_pred*y_true, axes)       
    #print(union.shape)
    temp = inter*1.0/(union+epsilon)    
    return temp
'''


class Metrics:
    #class for calculation of all metrics for each image, so as to make more elegant and save effort   
    #batchwise => not a mean value, calculated across batch

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
    
    

        

