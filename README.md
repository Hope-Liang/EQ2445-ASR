# EQ2445-ASR
The code repo for EQ2445 - Fall22 - Speech (Phoneme) Recognition <br/>
Team Members: Ernan Wang, Jingwen Fu, Jingxuan Mao, Sushanth Patil, Xinyu Liang

## Usage
Put the **TIMIT** data folder in parallel with this folder. <br/>
Put the **NoiseX_16kHz** data folder in parallel with this folder. <br/>

### Spectrogram Analysis


### Data Preprocessing
Use the **data_preprocessing.py** file. <br/>
You can extract 39-dim MFCC features and 123-dim filter-bank features from the TIMIT raw dataset. The way to select is by setting `feature = "MFCC39"` or `feature = "FilterBank123"`. <br/>
You can choose to encode the labels into 0-38 or 0-61 by setting `nPhonemes = 39` or `nPhonemes = 61`. 

### Model Training
Use the **data_training.py** file. <br/>
The code is designed for running on local devices, while a switch for running on Google Colab is also avalible if setting `GDrive = True`. If training on Colab, please create a folder **Data** on Drive and upload the preprocessed **pkl** files in it. <br/>
Several advanced settings available including: <br/>
i) `save_model` to specify whether to save the best model during each run. <br/>
ii) `feature` and `nPhonemes` defined the same way as in preprocessing part. <br/>
iii) `seq_len` is 800 by default as the maximum sequence length in TIMIT dataset is 777 in our way of feature extractions. <br/>
iv) `batch_size` and `n_epochs`. <br/>
v) `input_noise` if want to add Gaussian(mean=0, var=input_noise) to inputs during training as a way of regularisation, by default set to 0 as adding no noise. <br/>
vi) `early_stop` if using early stop regularisation technique, by default set to True. In this way it works together with `patience` for specifying after how many epochs it stops training after validation_acc not decreasing any more.


### Model Performance
