# Neural encoding with visual attention 

Source code for the following paper: https://arxiv.org/pdf/2010.00516.pdf (To appear in the proceedings of NeurIPS 2020)

# Background
In the present study, we propose a novel approach to neural encoding by including a trainable soft-attention module. Using our new approach, we demonstrate
that it is possible to learn visual attention policies by end-to-end learning merely on fMRI response data, and without relying on any eye-tracking. We showcase our approach on high-resolution 7T fMRI data from the Human Connectome Project movie-watching protocol and demonstrate the benefits of attentional masking in neural encoding. Importantly, we also find that attention locations estimated by the model on independent data agree well with the corresponding eye fixation patterns, despite no explicit supervision to do so.

# Getting started
_Data organization_ <br>
All experiments in this study are based on the Human Connectome Project movie-watching database. The dataset is publically available for download through the ConnectomeDB software [https://db.humanconnectome.org/]. Here, we utilized 7T fMRI data from the 'Movie Task fMRI 1.6mm/59k FIX-Denoised' package. Training models using the code provided herein will be easiest if data is organized according to the file structure within the data folder of this repo. Once the data is downloaded, run "preprocess_fMRI.py --movie #index" to compute the mean activations across all subjects and normalize the range of fMRI data for all 4 movies (index 1-4)

_Training_ <br>
To train the neural encoding model with learnable attention, run the following script from the scripts folder: <br>
python train_attention.py --lrate 0.0001 --epochs 50 --data_dir /path/to/preprocess/files --model_file /path/to/model --log_file /path/to/log --delay 4 --gpu_device 0 --batch_size 1

_Dependencies_ <br>
The code has been tested with following package versions in python: <br>
NumPy = 1.15.4 <br>
Tensorflow = 1.12.0 <br>
Keras = 2.2.4 <br>
Nibabel = 2.4.1 <br>


# References
* Meenakshi Khosla, Gia H. Ngo, Keith Jamison, Amy Kuceyeski and Mert R. Sabuncu. Neural encoding with visual attention. Tech report, arXiv, October 2020. 

# Bugs and Questions 
Please contact Meenakshi Khosla at mk2299@cornell.edu if you have any questions.  


