# EQ2445-ASR
The code repo for EQ2445 - Fall22 - Speech (Phoneme) Recognition <br/>
Team Members: Ernan Wang, Jingwen Fu, Jingxuan Mao, Sushanth Patil, Xinyu Liang

## Usage
Put the **TIMIT** data folder in parallel with this folder. <br/>
Put the **NOISEX92_RawDataset** data folder in parallel with this folder. <br/>

### Spectrogram Analysis


### Data Preprocessing
Use the **data_preprocessing.py** file. <br/>
0. You can extract 39-dim MFCC features and 123-dim filter-bank features from the TIMIT raw dataset. The way to select is by setting `feature = "MFCC39"` or `feature = "FilterBank123"`. <br/>
1. You can choose to encode the labels into 0-38 or 0-61 by setting `nPhonemes = 39` or `nPhonemes = 61`. <br/>
2. You can choose to add noise or not when preprocessing the data, by default `add_noise = False` and you will have a file containing training-evaluation-test data, and SNR and noise_type won't take effect. If setting `add_noise = True`, you only get a file containing the test data, and `SNR` and `noise_type` have to be specified.

### Model Training
Use the **data_training.py** file. <br/>
0. The code is designed for running on local devices, while a switch for running on Google Colab is also avalible if setting `GDrive = True`. If training on Colab, please create a folder **Data** on Drive and upload the preprocessed **pkl** files in it. <br/>
Several advanced settings available including: <br/>
1. `save_model`: specify whether to save the best model during each run, by default set to `True`. <br/>
2. `feature`: use "MFCC39" or "FilterBank123". <br/>
3. `nPhonemes_read`: the pkl file to read from has encoded the labels to 61 or 39 phonemes, refer to `nPhonemes` in preprocessing. <br/>
4. `nPhonemes_train`: trained with how many phonemes, e.g. the network output dimension size. <br/>
5. `nPhonemes_eval`: evaluate with how many phonemes, refering to performance measurement. <br/>
6. `input_noise` if want to add Gaussian(mean=0, std=input_noise) to inputs during training as a way of regularisation, by default set to `0` as adding no noise. <br/>
7. `weight_noise` if want to add Gaussian(mean=0, std=weight_noise) to weights during training as a way of regularisation, by default set to `0` as adding no noise. <br/>
8. `early_stop` if using early stop regularisation technique, by default set to `True`. In this way it works together with `patience` for specifying after how many epochs it stops training after validation_acc not decreasing any more, which is by default set to `8`.

### Model Evaluation
Use the **data_testing.py** file. <br/>


### Model Performance
Please update and refer to this table. [Google Sheet](https://docs.google.com/spreadsheets/d/1aCmCV1JPraFxDr_IoP4n-uptovXWFF9bb_pnxz_A7Bo/edit?usp=sharing)

