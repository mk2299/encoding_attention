import numpy as np
import keras 
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.applications.resnet50 import ResNet50


def gauss2D(shape=(5,5),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def pretrained_learnable(pretrained = True, input_shape = (23, 32, 2048)):
            
    
   
    input_layer =  Input(shape = input_shape, dtype='float32')
     
    intermediate = Conv2D(1, kernel_size = 5, padding = "same", activation = 'relu', name='l1connectordense')(input_layer)
    
    in_channels = 1  # the number of input channels
    kernel_size = 5 
    kernel_weights = gauss2D(shape = (kernel_size, kernel_size))
    
    ### Smoothing the output attention map 
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1) # apply the same filter on all the input channels
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    
    g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    g_layer_out = g_layer(intermediate)  # apply it on the input Tensor of this layer

    g_layer.set_weights([kernel_weights])
    g_layer.trainable = False 

    #xx, yy = intermediate.shape[1:3]
    intermediate = Flatten(name='flatc1')(g_layer_out)  
    intermediate = Activation('softmax', name='softmax1')(intermediate)  
    
    attention_1 = Reshape((*input_shape[:2],1), name='reshape1')(intermediate)  
    
    ### Apply attentional masking 
    layer4 = Multiply()([input_layer, attention_1])
    
    z = GlobalAveragePooling2D()(layer4) 
    z = Dense(1024, activation = 'elu')(z)    
    z = Dense(6*7*6*1024, activation='elu')(z)
    
    ## Response model starts here 
    y = Reshape((6,7,6,1024))(z)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = input_layer , outputs = y)
    return model 




