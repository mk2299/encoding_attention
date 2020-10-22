import os

import argparse

          
def train():
    parser = argparse.ArgumentParser(description='Learnable attention model')
    parser.add_argument('--data_dir', default="/home/mk2299/HCP_Movies/HCP_ConnectomeDB_Revised/preprocess/", type=str, help='Data directory')
    parser.add_argument('--lrate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default = 4, type=int)
    parser.add_argument('--model_file', default = None, help = 'Location for saving model')
    parser.add_argument('--log_file', default = None, help = 'Location for saving logs')
    parser.add_argument('--gpu_devices', default = "1,2", type = str, help = 'Device IDs')
    parser.add_argument('--gpu_count', default = None, type =int, help = 'Device count')
    parser.add_argument('--pretrained', default = 1, type = int, help = 'Freeze ResNet weights')
    parser.add_argument('--delay', default = None, type = int, help = 'HR') ## Set to 4sec
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    import numpy as np

    from models.models import learnable_attention
    from dataloader_gaze_final import DataGenerator_nogaze
    from keras.callbacks import ModelCheckpoint, EarlyStopping,  CSVLogger
    from keras import optimizers
    from keras.utils import multi_gpu_model
    from keras.models import load_model, Model
    from keras import optimizers
    from utils.losses import LossHistory
    import keras 
    
    
    class CustomSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if epoch % 4 == 0:  # or save after some epoch, each k-th epoch etc.
                self.model.save(args.model_file + "-{}.h5".format(epoch)) #"model_{}.h5".format(epoch))
            
    
    IDs_train = np.genfromtxt(os.path.join(args.data_dir, 'ListIDs_mean_train.txt'), dtype = 'str') 
    IDs_val = np.genfromtxt(os.path.join(args.data_dir,'ListIDs_mean_val.txt'), dtype = 'str') 
    
    train_generator = DataGenerator_nogaze(IDs_train,  batch_size = args.batch_size, train = True, delay = args.delay)
    val_generator = DataGenerator_nogaze(IDs_val,  batch_size = args.batch_size, train = False, delay = args.delay)

    history = LossHistory()
    callback_save = CustomSaver() 
    saver = CSVLogger(args.log_file)
    
    model = learnable_attention()
    if args.gpu_count>1:
        model = multi_gpu_model(model, gpus = args.gpu_count)
    print(model.summary())
    model.compile(optimizer=optimizers.Adam(lr=args.lrate, amsgrad=True), loss='mean_squared_error',metrics=['mean_squared_error'])
    model.fit_generator(
        train_generator,
        validation_data=val_generator,
        callbacks = [history, saver, callback_save], steps_per_epoch = 1000, validation_steps = 100, 
        epochs = args.epochs)

if __name__ == '__main__':
    train()
