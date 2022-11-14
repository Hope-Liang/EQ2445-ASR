# EQ2445-ASR
The code repo for EQ2445 - Fall22 - Speech (Phoneme) Recognition
Team Members: Ernan Wang, Jingwen Fu, Jingxuan Mao, Sushanth Patil, Xinyu Liang

## Usage
Put the **TIMIT** data folder in parallel with this folder. <br/>
Put the **NoiseX_16kHz** data folder in parallel with this folder.

### Spectrogram Analysis

### Data Preprocessing
Use the **data_preprocessing.py** file. <br/>
You can extract 39-dim MFCC features and 123-dim filter-bank features from the TIMIT raw dataset. The way to select is by setting `feature = "MFCC39"` or `feature = "FilterBank123"`. <br/>
You can choose to encode the labels into 0-38 or 0-61 by setting `nPhonemes = 39` or `nPhonemes = 61`. 

### Model Training
For the `data_training.py` file, you can choose to run it on Google Colab by reading data from Google Drive (in this way if you create a folder called `Data` and put the preprocessed `pkl` files into it, everything shall work) by setting `GDrive = True` and locally if `False`. You only need to adjust the settings and hyperparameters part in this implementation.