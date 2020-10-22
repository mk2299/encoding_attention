import numpy as np
import keras
import imageio
from skimage.transform import resize
import nibabel as nib
import os
import tensorflow as tf

class DataGenerator_nogaze(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size = 1, fraction = 0, dim=(720,1024),  volume = (111, 127, 111), n_channels = 3, delay = None, root_stimuli = '/share/sablab/nfs02/data/HCP_movie/Post_20140821_version/', root_brain = '/home/mk2299/HCP_Movies/HCP_ConnectomeDB_Revised/preprocess/', shuffle=True):
        'Initialization'
        
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']

        self.fraction = fraction
        self.root_data = root_stimuli
        self.vol_root = root_brain
        
        self.delay = delay
     
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path,  'ffmpeg'))
            
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
            
            X[i] = np.array(self.videos[int(movie)-1].get_data(int(frame)-12*self.fraction))
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root,  'MOVIE'+ movie+'_mean_train.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay] 

        return X, y    
    
    
class DataGenerator_gaze(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size = 1, fraction = 0, dim=(720,1024),  volume = (111, 127, 111), n_channels = 3, delay = None, shuffle=True, path_gaze_attention = '/home/mk2299/HCP_Movies/HCP_ConnectomeDB_Revised/eyegaze/preprocess/corrected/visual_att_weights_resized.npy', root_stimuli = '/share/sablab/nfs02/data/HCP_movie/Post_20140821_version/', root_brain = '/home/mk2299/HCP_Movies/HCP_ConnectomeDB_Revised/preprocess/', layer_sizes = [(1, 180, 256, 256), (1, 90, 128, 256), (1, 45, 64, 256), (1, 23, 32, 2048)]):
        'Initialization'
        
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']

        self.fraction = fraction
        self.root_data= root_stimuli
        self.vol_root = root_brain 
        
        self.delay = delay
        self.layer_sizes = layer_sizes
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path,  'ffmpeg'))
            
        self.dim = dim
        self.volume = volume 
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs
        self.filt_imgs = np.load(path_gaze_attention).item()
        
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
            weights = self.filt_imgs[movie][int(int(frame)/24)][3]
            weights = weights/np.sum(weights)
            att_weights = np.tile(weights[np.newaxis, :, :, np.newaxis], (1, 1, 1, self.layer_sizes[3][-1]))
            X[i] = np.array(self.videos[int(movie)-1].get_data(int(frame)))
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root,  'MOVIE'+ movie+'_mean_train.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay] 

        return [X, att_weights], y       