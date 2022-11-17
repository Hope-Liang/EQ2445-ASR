import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import os
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from six.moves import cPickle
from google.colab import drive
from torch.utils.data import TensorDataset, DataLoader
import pickle5 as pkl5


" ***************************************** Loading data  *********************************************"
# all data are of 61 phonemes and already normalized
# Need to change the path of files
# MFCC39
with open("../TIMIT_MFCC39_nPhonemes61_clean.pkl", 'rb') as f:
    x_train, y_train, x_val, y_val, x_test, y_test = pkl5.load(f)

# Fourier FilterBank123
with open("../TIMIT_FilterBank123_nPhonemes61_clean.pkl", 'rb') as f:
    x_train, y_train, x_val, y_val, x_test, y_test = pkl5.load(f)



" ****************************************** Parameters ***************************************** "
# Parameters for data preprocessing
feature = "MFCC39"            # "FilterBank123" or "MFCC39"
DIM = 39                     # 123 for FilterBank123 and 39 for MFCC39
nPhoneme_train = 61           # 39 or 61
nPhoneme_pred = 39            # 39 or 61
seq_len = 800                 # padded sequence length, longest sequence of length 777
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Hyperparameters
batch_size = 128          # batch size, 128 for MFCC, 64 for FilterBank123
n_epochs = 50             # number of training epochs


# Regularisations
input_noise = 0.5       # std of noise added to input x during training
weight_noise = 0          # std of noise added to weights during training
dropout = 0               # 0 if no dropout added
early_stop = True        # early stop or not
sch_LR = False            # Updating LR using scheduler or not
patience = 8              # hyper-parameter for early stopping


# Setting for saving model
save_model = True
# Need to change the path of files
savemodelname = "../ASR_Data/best_regu_"+feature+"_train"+str(nPhoneme_train)+"_pred"+str(nPhoneme_pred)+".pt"



" ****************************************** Data Processing ***************************************** "
# Mapping if only 39 phonemes need to be trained and predicted
# Adding mapping in training loop will result in longer training and evaluation time
# So mapping before training can save time
phoneme_set_61_39 = {
    'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 
    'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',
    'eng': 'ng', 'zh': 'sh', "ux": "uw", "pcl": "sil", "tcl": "sil", 
    "kcl": "sil", "qcl": "sil", "bcl": "sil", "dcl": "sil", "gcl": "sil",
    "h#": "sil", "#h": "sil", "pau": "sil", "epi": "sil", "q": "sil",
}
phoneme_set_39_list = [
    'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow',  # 13 phns
    'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx',  # 14 phns
    'g', 'p', 't', 'k', 'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil'  # 12 pns
]
phoneme_set_61_list = [
    'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr',
    'ax-h', 'jh',
    'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'nx',
    'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi',
    'h#',
]

# phoneme_set_61 = dict(zip(phoneme_set_61_list, list(range(61))))
phn_to_ix_39 = dict(zip(phoneme_set_39_list, list(range(39))))
ix_to_phn_39 = dict((v, k) for k, v in phn_to_ix_39.items())


def map_61_39_and_vectorize(y): # the input should be y_train, y_val, y_test
    y_ret = []
    for i in range(len(y)):
        j = 0
        y_temp = np.empty(y[i].shape)
        for phn in y[i]:
            phn = phoneme_set_61_list[phn]
            if phn in phoneme_set_61_39.keys():
                phn = phoneme_set_61_39[phn]
            y_temp[j] = phn_to_ix_39[phn]
            j += 1
        y_ret.append(y_temp)
    return y_ret


def set_type(X, type):
    for i in range(len(X)):
        X[i] = X[i].astype(type)
    return X


if nPhoneme_train = 39 and nPhoneme_pred == 39:
	y_train = set_type(y_train, int)
	y_val = set_type(y_val, int)
	y_test = set_type(y_test, int)

	y_train = map_61_39_and_vectorize(y_train)
	y_val = map_61_39_and_vectorize(y_val)
	y_test = map_61_39_and_vectorize(y_test)



" **************************************** Padding and Set data loader ************************************** "
def paddingX(X, ttl_length):
    X_padded = []
    for i in range(len(X)):
        x = torch.Tensor(X[i])
        x_padded = F.pad(x, (0,0,0,ttl_length-x.shape[0]))
        X_padded.append(x_padded)
    return X_padded


def paddingY(Y, ttl_length):
    Y_padded = []
    for i in range(len(Y)):
        y = torch.Tensor(Y[i])
        y_padded = F.pad(y, (0,ttl_length-y.shape[0]), mode='constant', value=-1)
        Y_padded.append(y_padded.type(torch.int64))
    return Y_padded


# x padded is a list of Tensors with size (seq_len, dim)
x_train_padded = paddingX(x_train, seq_len) 
x_val_padded = paddingX(x_val, seq_len)
x_test_padded = paddingX(x_test, seq_len)
# y padded is a list of Tensors with size (seq_len), padded mark is -1
y_train_padded = paddingY(y_train, seq_len) 
y_val_padded = paddingY(y_val, seq_len)
y_test_padded = paddingY(y_test, seq_len)
# turning x into a 3-D tensor
x_train = torch.cat(x_train_padded, 0).reshape(-1,seq_len,DIM) # shape(4158, 800, dim)
x_val = torch.cat(x_val_padded, 0).reshape(-1,seq_len,DIM)
x_test = torch.cat(x_test_padded, 0).reshape(-1,seq_len,DIM)
# turning y into a 2-D tensor
y_train = torch.cat(y_train_padded, 0).reshape(-1,seq_len) # shape (4158,800)
y_val = torch.cat(y_val_padded, 0).reshape(-1,seq_len)
y_test = torch.cat(y_test_padded, 0).reshape(-1,seq_len)

# Data loader
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
test_data = TensorDataset(x_test, y_test)
# shuffle data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)



" ******************************************* Model Construction ************************************** "
def add_noise(weights, noise_w): 
	'''
	Add noise to training weights
	'''                                                                                                                                                                                                                                             
    with torch.no_grad(): 
      noise = torch.normal(0, noise_w, weights.size())                                                                                                                                                                                                                                                  
      weight_noise = nn.Parameter(weights + noise.to(weights.device))                                                                                                                                                                                                                                                                                                                                                                                                                                     
    return weight_noise


def map_61_39(y):
	'''
	Map the output of network from 61 phonemes to 39 phonemes
	''' 
    y_mapped = []
    for phn in y:
      phn = phoneme_set_61_list[phn]
      if phn in phoneme_set_61_39.keys():
        phn = phoneme_set_61_39[phn]
      y_mapped.append(phn_to_ix_39[phn])
    y_mapped = torch.tensor(y_mapped).reshape(-1, 1)
    return y_mapped



class LSTMASR(nn.Module):
    '''
    PARAMETERS:
    in_dim: The dimension of features in the input x, 39 or 61
    out_dim: The dimension of features in the output y, 39 or 61, 
    noise_x: The std of the gaussian noise added to the input x in training
    noise_w: The std of the gaussian noise added to the weights in training
    dropout: Dropout layer
    '''
    def __init__(self, in_dim=39, out_dim=39, noise_x=0, noise_w=0, dropout=0):
        super(LSTMASR, self).__init__()
        self.input_size = in_dim # input_size = The dimension of features in the input x
        self.hidden_size = 250 # hidden_size = The number of features in the hidden state h, following Grave's paper we use 250
        self.output_size = out_dim # The number of phonemes contained in output, 39 or 61
        #self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, bias=False, batch_first = True, bidirectional=True, dropout = dropout)
        ## bias = False following other's implementation, double-check
        ## batch_first = True, the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.out = nn.Linear(self.hidden_size*2, self.output_size) # add a fully connected layer
        #self.weight_init()
        self.noise_x = noise_x
        self.noise_w = noise_w

    def weight_init(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'bias' not in name:
                        torch.nn.init.orthogonal_(param.data)

    def forward(self, X):
        if self.training:
            # Adding noise to the training data for regularization
            X += torch.normal(0, self.noise_x, X.size()).to(X.device)
            # Adding noise to the training weights for regularization
            self.lstm.weight_hh_l0 = add_noise(self.lstm.weight_hh_l0, self.noise_w)
            self.lstm.weight_hh_l1 = add_noise(self.lstm.weight_hh_l1, self.noise_w)
        lstm_out, _ = self.lstm(X) # lstm_out is the the output features from the last layer of the LSTM, for each token
        linear_out = self.out(lstm_out)
        return linear_out # need to apply a softmax to generate the probabilities


def train_one_epoch(m, optimizer, dataloader, device, nPhoneme_train, nPhoneme_pred):
    '''
    Function for training the network for one epoch
    m: training model
    dataloader: val_loader for validation, test_loader for testing
    device: GPU or CPU
    nPhoneme_train: The number of phonemes in training (39 or 61), dimension of output
    nPhoneme_pred: The number of phonemes needs to be predicted (39 or 61)
    '''
    # Set model to trainning mode
    m.train()

    # Initialization for calcualtion of acc
    corrects = 0
    ttl = 0

    for step, (x,y) in enumerate(dataloader):
        # Push params to the device (GPU/CPU)
        x = x.to(device)
        y = y.to(device)

        # Clean accumulated gradients
        optimizer.zero_grad()

        # Add gaussian noise to the model 
        # m.apply(add_noise_to_weights)

        # Get model output (N, S, D)
        y_pred = m(x)

        # Mask out the meaningful output of the model by ruling out padding part
        mask = y.ge(0)
        mask_expanded = mask.unsqueeze(-1).expand(-1,-1,nPhoneme_train)
        y = torch.masked_select(y,mask)
        y_pred = torch.masked_select(y_pred,mask_expanded).reshape(-1,nPhoneme_train)

        # Add softmax layer for prediction to transfer model output to logistics
        y_soft = F.softmax(y_pred, dim=1)
        ttl += len(y)
        if nPhoneme_pred == 39 and nPhoneme_train == 61:
            y_pred_o = torch.max(y_soft,1)[1].detach().cpu() # The dimension of y_pred_o can be 39 or 61 when predict 39
            y_pred39 = map_61_39(y_pred_o)
            y39 = map_61_39(y)
            corrects += (y_pred39 == y39).sum().item()
        else:
            corrects += (torch.max(y_soft,1)[1] == y ).detach().cpu().sum().item()  # The dimension of y_pred can only be 61 when predict 61

        # Calculate loss and update weights
        loss = nn.CrossEntropyLoss()(y_pred, y)
        if step%10 ==0:   # Print training loss every 10 mini-batches
            print(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()

    return loss.detach().cpu().item(), corrects/ttl


def evaluate(m, dataloader, device, nPhoneme_train, nPhoneme_pred):
    '''
    Function for Model evaluation
    m: trained model
    dataloader: val_loader for validation, test_loader for testing
    device: GPU or CPU
    nPhoneme_train: The number of phonemes in training (39 or 61), dimension of output
    nPhoneme_pred: The number of phonemes needs to be predicted (39 or 61)
    '''
    m.eval() # set model to evaluation mode
    corrects = 0
    ttl = 0
    with torch.no_grad(): # Stop updating weights
        for step, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            y_out = m(x)
            mask = y.ge(0)
            mask_expanded = mask.unsqueeze(-1).expand(-1,-1,nPhoneme_train)

            y = torch.masked_select(y,mask)  # mask out the meaningless part of labels, y is a 1D Tensor of ground truth Phoneme IDs as a long sequence
            y_out = torch.masked_select(y_out,mask_expanded).reshape(-1,nPhoneme_train)

            # Add softmax layer for prediction to transfer model output to logistics
            y_soft = F.softmax(y_out, dim=1)

            ttl += len(y)
            if nPhoneme_pred == 39 and nPhoneme_train == 61:
                y_pred_o = torch.max(y_soft,1)[1].detach().cpu()
                y_pred39 = map_61_39(y_pred_o)
                y39 = map_61_39(y)
                corrects += (y_pred39 == y39).sum().item()
            else:
                corrects += (torch.max(y_soft,1)[1] == y ).detach().cpu().sum().item()

            loss = nn.CrossEntropyLoss()(y_out, y)

    return loss.detach().cpu().item(), corrects/ttl



def train_model(m, train_loader, val_loader, opt, device, nPhoneme_train, nPhoneme_pred, n_epochs, early_stop=False, sch_LR=False, save_model=False, savemodelname=None):
    '''
    Function for training and evaluating the network by repeatly calling train_one_epoch() and evaluation()
    '''
    print("++++++++++++++++Training Start++++++++++++++++++")
    # print model information
    print("Model's state_dict:")
    for param_tensor in m.state_dict():
        print(param_tensor, "\t", m.state_dict()[param_tensor].size())
    train_loss = []
    val_loss = []
    best = 1e8
    for i in range(n_epochs):
        print("=================Epoch {}==================".format(i))
        # Training
        train_loss_i, train_acc_i = train_one_epoch(m,opt,train_loader,device, nPhoneme_train, nPhoneme_pred)
        print("training loss:", train_loss_i)
        print("training acc:", train_acc_i)
        # Validation
        val_loss_i, val_acc_i = evaluate(m,val_loader,device, nPhoneme_train, nPhoneme_pred)
        print("validation loss:", val_loss_i)
        print("validation acc:", val_acc_i)
        train_loss.append(train_loss_i)
        val_loss.append(val_loss_i) 
        # Adjust learning rate
        if sch_LR:
            scheduler.step()
        # Save best model and early stop           
        if val_loss_i < best:
            best = val_loss_i
            if save_model:
                torch.save(m.to('cpu').state_dict(), savemodelname)
                m.to(device)
            if early_stop:
                p = patience
        else:
            if early_stop:
                p -= 1
                if p < 1:
                    break
    return train_loss, val_loss



" ******************************************** Model Training and Testing ************************************** "
# Define model
m = LSTMASR(in_dim=DIM, out_dim=nPhoneme_train, noise_x=input_noise, noise_w=weight_noise, dropout=dropout)
m.to(device)
opt = torch.optim.Adam(m.parameters(), lr=0.001, betas=(.9,.99))
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.8)


# Train the model and save the best model
train_loss, val_loss = train_model(m, train_loader, val_loader, opt, device, 
                                   nPhoneme_train, nPhoneme_pred, n_epochs, 
                                   early_stop, sch_LR, save_model, savemodelname)



# Test the best model
print("++++++++++++++++Testing Start++++++++++++++++++")
m_best = LSTMASR(in_dim=DIM, out_dim=nPhoneme_train, noise_x=input_noise, noise_w=weight_noise, dropout=dropout)
m_best.load_state_dict(torch.load(savemodelname))
m_best.eval()
m_best.to(device)
_, test_acc = evaluate(m_best, test_loader, device, nPhoneme_train, nPhoneme_pred)
print("test acc:", test_acc)
