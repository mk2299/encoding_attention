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

def learnable_attention(freeze = True, input_shape = (720, 1024, 3)):
    
    base_model = ResNet50(include_top= False, weights="imagenet", input_shape= input_shape)
    if freeze:
           for layer in base_model.layers:
                   layer.trainable = False
                    
    base_model.layers.pop()
    base_model.layers[-1].outbound_nodes = []
    base_model.outputs = [base_model.layers[-1].output]                
    
   ######### Get last layer output 

    layer4 =  base_model.get_output_at(-1)
     
    intermediate = Conv2D(1, kernel_size = 5, padding = "same", activation = 'relu', name='l1connectordense')(layer4)
    
    in_channels = 1  # the number of input channels
    kernel_size = 5 
    kernel_weights = gauss2D(shape = (kernel_size, kernel_size))
    
    
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1) # apply the same filter on all the input channels
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    
    g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    g_layer_out = g_layer(intermediate)  # apply it on the input Tensor of this layer

    g_layer.set_weights([kernel_weights])
    g_layer.trainable = False 

    #xx, yy = intermediate.shape[1:3]
    intermediate = Flatten(name='flatc1')(g_layer_out)  # batch*xy
    intermediate = Activation('softmax', name='softmax1')(intermediate)  # batch*xy
    
    #### Hard coded for input shape (720, 1024, 3) 
    attention_1 = Reshape((23,32,1), name='reshape1')(intermediate)  # batch*xy*512.

    layer4 = Multiply()([layer4, attention_1])
    
    z = GlobalAveragePooling2D()(layer4) 
    z = Dense(1024, activation = 'elu')(z)    
    z = Dense(6*7*6*1024, activation='elu')(z)
    ### Response model starts here 
    y = Reshape((6,7,6,1024))(z)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = base_model.input , outputs = y)
    return model 

def uniform_attention(freeze = True, input_shape = (720, 1024, 3)):
  
    
    base_model = ResNet50(include_top= False, weights="imagenet", input_shape=input_shape)
    if freeze:
        for layer in base_model.layers:
               layer.trainable = False
                    
    
    
    layer4 =  GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='elu')(layer4)    
    x = Dense(6*7*6*1024, activation='elu')(x)
    
    #### Response model starts here
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = base_model.input , outputs = y)
    return model


def gaze_attention(freeze = True, layer_size = (23, 32, 2048), input_shape=(720,1024,3)):
    
  
    base_model = ResNet50(include_top= False, weights="imagenet", input_shape= input_shape)
    if freeze:
           for layer in base_model.layers:
                   layer.trainable = False
                    
    base_model.layers.pop()
    base_model.layers[-1].outbound_nodes = []
    base_model.outputs = [base_model.layers[-1].output]                

    att_weights4 = Input(shape = layer_size, dtype='float32', name='attention_gaze')
    layer4 =  base_model.get_output_at(-1)    
    weight_layer = Multiply()([layer4, att_weights4])
    
    z = GlobalAveragePooling2D()(weight_layer) 
    z = Dense(1024, activation = 'elu')(z)    
    z = Dense(6*7*6*1024, activation='elu')(z)
    
    #### Response model starts here 
    y = Reshape((6,7,6,1024))(z)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = [base_model.input, att_weights4], outputs = y)
    return model 



def no_attention(freeze = True, input_shape=(720,1024,3)):
  
    
    base_model = ResNet50(include_top= False, weights="imagenet", input_shape=input_shape)
    if freeze:
        for layer in base_model.layers:
               layer.trainable = False
                        
    
    layer4 =  Flatten()(base_model.output)
    x = Dense(256, activation='elu')(layer4)    
    x = Dense(6*7*6*1024, activation='elu')(x)
    
    ### Response model starts here 
    
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = base_model.input , outputs = y)
    return model