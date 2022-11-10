import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle5 as pickle

# a map from 61 phonemes to 39 phonemes
phoneme_map_61_39 = {
    'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm', 'en': 'n', 
    'nx': 'n', 'eng': 'ng', 'zh': 'sh', "ux": "uw", "pcl": "sil", "tcl": "sil", "kcl": "sil", "qcl": "sil", 
    "bcl": "sil", "dcl": "sil", "gcl": "sil", "h#": "sil", "#h": "sil", "pau": "sil", "epi": "sil", "q": "sil"
}
# a list of the 39 phonemes, check https://www.researchgate.net/publication/275055833_TCD-TIMIT_An_audio-visual_corpus_of_continuous_speech
phoneme_set_39_list = [
    'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow', 'l', 'r', 'y', 'w', 'er', 'm', 
    'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k', 'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil'
]
# a list of the 61 phonemes, check http://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database, page 5
phoneme_set_61_list = [
    'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr','ax-h', 
    'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng',  'em', 'nx', 
    'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi','h#'
]
# a dictionary to turn phonemes from the 61 set into number 0 to 60
phn61_to_val_dict = dict(zip(phoneme_set_61_list, list(range(61))))
# a dictionary to turn phonemes from the 39 set into number 0 to 38
phn39_to_val_dict = dict(zip(phoneme_set_39_list, list(range(39))))
# a dictionary to turn number 0 to 60 into a phoneme in the 61 set
val_to_phn61_dict = dict((v, k) for k, v in phn61_to_val_dict.items()) 
# a dictionary to turn number 0 to 38 into a phoneme in the 39 set
val_to_phn39_dict = dict((v, k) for k, v in phn39_to_val_dict.items())

def paddingX(X, ttl_length):
    '''
    input: X is a list of 2D feature arrays with different lengths, 
           ttl_length is the total length for each array to be padded to, should be larger than the longest array
    output: a list of 2D padded feature arrays, padded with 0 at the end
    '''
    X_padded = []
    for i in range(len(X)):
        x = torch.Tensor(X[i])
        x_padded = F.pad(x, (0,0,0,ttl_length-x.shape[0]))
        X_padded.append(x_padded)
    return X_padded

def paddingy(Y, ttl_length):
    '''
    input: Y is a list of 1D label arrays with different lengths, 
           ttl_length is the total length for each array to be padded to, should be larger than the longest array
    output: a list of 1D padded label arrays, padded with -1 at the end
    '''
    Y_padded = []
    for i in range(len(Y)):
        y = torch.Tensor(Y[i])
        y_padded = F.pad(y, (0,ttl_length-y.shape[0]), mode='constant', value=-1)
        Y_padded.append(y_padded.type(torch.int64))
    return Y_padded

def onehotY(Y, nPhonemes):
    '''
    input: Y is a list of 1D label arrays with different lengths
           nPhonemes is the number of classes the label corresponds to
    output: a list of one-hot encoded 2D label arrays
    '''
    Y_onehot = []
    for i in range(len(Y)):
        y = torch.Tensor(Y[i]).type(torch.int64)
        y_onehot = F.one_hot(y, num_classes = nPhonemes)
        Y_onehot.append(y_onehot)
    return Y_onehot

def paddingY(Y, ttl_length):
    '''
    input: Y is a list of 2D one-hot encoded label arrays with different lengths, 
           ttl_length is the total length for each array to be padded to, should be larger than the longest array
    output: a list of 2D padded one-hot encoded label arrays, padded with -1 at the end
    '''
    Y_padded = []
    for i in range(len(Y)):
        y = torch.Tensor(Y[i])
        y_padded = F.pad(y, (0,0,0,ttl_length-y.shape[0]), mode='constant', value=-1)
        Y_padded.append(y_padded)
    return Y_padded


class LSTMASR(nn.Module):

    def __init__(self, noise=0, dropout=0):
        super(LSTMASR, self).__init__()
        self.input_size = DIM # input_size = The dimension of features in the input x
        self.hidden_size = 250 # hidden_size = The number of features in the hidden state h, following Grave's paper we use 250
        self.output_size = nPhonemes
        #self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, bias=False, batch_first = True, bidirectional=True, dropout = dropout)
        ## bias = False following other's implementation, double-check
        ## batch_first = True, the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.out = nn.Linear(self.hidden_size*2, self.output_size) # add a fully connected layer
        #self.weight_init()
        self.noise = noise

    def weight_init(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'bias' not in name:
                        torch.nn.init.orthogonal_(param.data)

    def forward(self, X):
        if self.training:
            X += torch.normal(0, self.noise, X.size()).to(X.device)
        lstm_out, _ = self.lstm(X) # lstm_out is the the output features from the last layer of the LSTM, for each token
        linear_out = self.out(lstm_out)
        return linear_out # need to apply a softmax to generate the probabilities

def cal_acc(m, X, y):
    y_pred = m(X)
    mask = y.ge(0)
    mask_expanded = mask.unsqueeze(-1).expand(-1,-1,nPhonemes)
    y = torch.masked_select(y,mask)
    y_pred = torch.masked_select(y_pred,mask_expanded).reshape(-1,nPhonemes)
    y_pred_idx = torch.max(y_pred,1)[1]
    return sum(y_pred_idx == y)/y.shape[0]

def cal_acc_intermediate(m, y, y_pred):
    y_pred_idx = torch.max(y_pred,1)[1]
    return sum(y_pred_idx == y)/y.shape[0]

def train_one_epoch(m, optimizer, dataloader, device, nPhonemes):
    m = m.train() # set model to training mode
    corrects = 0
    ttl = 0
    for step, (x,y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad() # clean accumulated gradients
        
        y_pred = m(x)
        mask = y.ge(0)
        mask_expanded = mask.unsqueeze(-1).expand(-1,-1,nPhonemes)

        y = torch.masked_select(y,mask) # mask out the meaningless part of labels
        y_pred = torch.masked_select(y_pred,mask_expanded).reshape(-1,nPhonemes) # mask out the meaningless part of predictions

        ttl += len(y)
        corrects += (torch.max(y_pred,1)[1] == y ).detach().cpu().sum().item()

        loss = nn.CrossEntropyLoss()(y_pred, y)
        #if step%10 ==0:
            #print(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()

    return loss.detach().cpu().item(), corrects/ttl

def evaluate(m, dataloader, device, nPhonemes):
    m = m.eval() # set model to evaluation mode
    corrects = 0
    ttl = 0
    with torch.no_grad():
        for step, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            y_pred = m(x)
            mask = y.ge(0)
            mask_expanded = mask.unsqueeze(-1).expand(-1,-1,nPhonemes)

            y = torch.masked_select(y,mask)
            y_pred = torch.masked_select(y_pred,mask_expanded).reshape(-1,nPhonemes)

            ttl += len(y)
            corrects += (torch.max(y_pred,1)[1] == y ).detach().cpu().sum().item()

            loss = nn.CrossEntropyLoss()(y_pred, y)

    return loss.detach().cpu().item(), corrects/ttl

def train_model(m, train_loader, val_loader, opt, device, nPhonemes, n_epochs, early_stop=False):
    print("++++++++++++++++Training Start++++++++++++++++++")
    train_loss = []
    val_loss = []
    best = 1e8
    for i in range(n_epochs):
        print("=================Epoch {}==================".format(i))
        train_loss_i, train_acc_i = train_one_epoch(m,opt,train_loader,device,nPhonemes)
        print("training loss:", train_loss_i)
        print("training acc:", train_acc_i)
        val_loss_i, val_acc_i = evaluate(m,val_loader,device,nPhonemes)
        print("validation loss:", val_loss_i)
        print("validation acc:", val_acc_i)
        train_loss.append(train_loss_i)
        val_loss.append(val_loss_i)
        if early_stop:
            if val_loss_i < best:
                best = val_loss_i
                p = patience
            else:
                p -= 1
            if p < 1:
                break
    return train_loss, val_loss


# Settings 
GDrive = False # state if reading from Google Drive or local folder
save_model = False
feature = "MFCC39" # "FilterBank123" or "MFCC39"
nPhonemes = 39 # 39 or 61
seq_len = 800 # padded sequence length, longest sequence of length 777
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device("mps") # for M1 Mac using GPU accelerator
# Hyperparameters
batch_size = 128 # batch size
n_epochs = 50 # number of training epochs
# Regularisations
input_noise = 0 # 0 if no noise added during training
dropout = 0 # 0 if no dropout added
early_stop = True
patience = 8 # hyper-parameter for early stopping


# read data
if GDrive:
    readname = "/content/gdrive/MyDrive/Data/TIMIT_"+feature+"_nPhonemes"+str(nPhonemes)+"_clean.pkl"
    savemodelname =  "/content/gdrive/MyDrive/Data/best.pt"
else:
    readname = "../TIMIT_"+feature+"_nPhonemes"+str(nPhonemes)+"_clean.pkl"
    savemodelname = "../best.pt"
with open(readname, 'rb') as f:
    data = pickle.load(f)
x_train, y_train, x_val, y_val, x_test, y_test = data

# pad data
x_train_padded = paddingX(x_train, seq_len) # x_train_padded is a list of Tensors with size (800, DIM)
x_val_padded = paddingX(x_val, seq_len)
x_test_padded = paddingX(x_test, seq_len)
y_train_padded = paddingy(y_train, seq_len) # y_train_padded is a list of Tensors with size (800,), padded mark is -1
y_val_padded = paddingy(y_val, seq_len)
y_test_padded = paddingy(y_test, seq_len)

# turn data to torch Tensor
if feature == "FilterBank123":
    DIM = 123
elif feature == "MFCC39":
    DIM = 39
else:
    print("ERROR: Unsupported feature!")
x_train = torch.cat(x_train_padded, 0).reshape(-1,800,DIM) # shape(4158, 800, DIM)
x_val = torch.cat(x_val_padded, 0).reshape(-1,800,DIM)
x_test = torch.cat(x_test_padded, 0).reshape(-1,800,DIM)
y_train = torch.cat(y_train_padded, 0).reshape(-1,800) # shape (4158, 800)
y_val = torch.cat(y_val_padded, 0).reshape(-1,800)
y_test = torch.cat(y_test_padded, 0).reshape(-1,800)

# turn data to TensorDataset
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
test_data = TensorDataset(x_test, y_test)

# shuffle data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

# define model
m = LSTMASR(noise=input_noise, dropout=dropout)
m.to(device)
opt = torch.optim.Adam(m.parameters(), lr=0.001, betas=(.9,.99))

# train and test model
train_loss, val_loss = train_model(m, train_loader, val_loader, opt, device, nPhonemes, n_epochs, early_stop)
print("++++++++++++++++Testing Start++++++++++++++++++")
_, test_acc = evaluate(m, test_loader, device, nPhonemes)
print("test acc:", test_acc)

# save_model
print("Model's state_dict:")
for param_tensor in m.state_dict():
    print(param_tensor, "\t", m.state_dict()[param_tensor].size())
if save_model:
    torch.save(m.to('cpu').state_dict(), savemodelname)
