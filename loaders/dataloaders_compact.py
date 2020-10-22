import numpy as np
import keras
import imageio
from skimage.transform import resize
import nibabel as nib
import os
import tensorflow as tf

class DataGenerator_nogaze(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size = 1, fraction = 0, dim=(23, 32),  volume = (111, 127, 111), n_channels = 2048, delay = None, root_stimuli = '/share/sablab/nfs02/data/HCP_movie/Post_20140821_version/', root_brain = '/home/mk2299/HCP_Movies/HCP_ConnectomeDB_Revised/preprocess/', shuffle=True):
        'Initialization'
        
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']

        self.fraction = fraction
        self.root_data= root_stimuli
        self.vol_root = root_brain
        
        self.delay = delay
     
        self.videos = []
        for m in ['1', '2', '3', '4']:
            self.videos.append(np.load('/home/mk2299/HCP_Movies/HCP_ConnectomeDB_Revised/eyegaze/preprocess/corrected/visual' + m + '_last_layer.npy'))  
            
        self.dim = dim
        self.volume = volume 
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs
        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]


        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.volume, 1))

        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            movie, frame = self.list_IDs[idx]
            
            X[i] = np.array(self.videos[int(movie)-1][int(int(frame)/24)])
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root,  'MOVIE'+ movie+'_mean_train.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay] ### Clipping to remove black borders, fps = 24 

        return X, y     