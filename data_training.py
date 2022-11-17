import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import os
import numpy as np
import pickle as pkl
import pickle5 as pkl5
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
#from google.colab import drive
#drive.mount('/content/gdrive')

############## Device and Other Settings ##############
GDrive = False # state if reading from Google Drive or local folder
save_model = True # check if automatically updates/saves the model coefficients
feature = "FilterBank123" # "FilterBank123" or "MFCC39"
seq_len = 800 # padded sequence length, longest sequence of length 777
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device("mps") # for M1 Mac using GPU accelerator


############## Hyperparameter Settings ################
# batch size and input feature dimension
if feature == "FilterBank123":
    batch_size = 64
    DIM = 123
elif feature == "MFCC39":
    batch_size = 128
    DIM = 39
else:
    print("ERROR: Feature {} not supported".format(feature))
nPhonemes_read = 61 # 39 or 61, shall be the same as nPhonemes_train to save memory, but could be different to save disk space
nPhonemes_train = 61 # 39 or 61
nPhonemes_eval = 39  # 39 or 61
assert nPhonemes_read >= nPhonemes_train
assert nPhonemes_eval <= nPhonemes_train
# number of training epochs
n_epochs = 50 


############## Optional Regularisations ################
input_noise = 0         # (IN) std of noise added to input x during training
weight_noise = 0        # (WN) std of noise added to weights during training
dropout = 0             # (DR) 0 if no dropout added
step_LR = False         # (SLR) (By default not used with Adam) StepLR used or not
step_size = 5           # learning rate decay step size
gamma = 0.8             # step size decay factor
early_stop = True       # (ES) (By default used) early stop or not
patience = 8            # hyper-parameter for early stopping


############## Mappings and Functions #################
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
# a dictionary to turn phonemeIDs from 0-60 to phonemeIDs from 0-38
val61_to_val39_dict = {}
for phn61 in phoneme_set_61_list:
    val61 = phn61_to_val_dict[phn61]
    if phn61 not in phoneme_set_39_list:
        phn39 = phoneme_map_61_39[phn61]
    else:
        phn39 = phn61
    val39 = phn39_to_val_dict[phn39]
    val61_to_val39_dict[val61]=val39

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

def mapfunc_val61_val39_list(y):
    '''
    input: a list of 1D numpy arrays of 61 labels (typically y_train )
    output: a list of 1D numpy arrays of 39 labels
    '''
    for i in range(len(y)):
        y[i] = y[i].astype(int)
    y_ret = []
    for i in range(len(y)):
        j = 0
        y_temp = np.empty(y[i].shape)
        for val61 in y[i]:
            val39 = val61_to_val39_dict[val61]
            y_temp[j] = val39
            j += 1
        y_ret.append(y_temp)
    return y_ret


############## Path Directing ##################
if GDrive:
    readname = "/content/gdrive/MyDrive/Data/TIMIT_"+feature+"_nPhonemes"+str(nPhonemes_read)+"_clean.pkl"
    savemodelname = "/content/gdrive/MyDrive/Data/best_"+feature+ \
    "_nPhonemes"+str(nPhonemes_train)+"-"+str(nPhonemes_eval)+ \
    "_IN"+str(input_noise)+"_WN"+str(weight_noise)+"_DR"+str(dropout)+ \
    "_ES"+".pt"
else:
    readname = "../TIMIT_"+feature+"_nPhonemes"+str(nPhonemes_read)+"_clean.pkl"
    savemodelname = "../best_"+feature+"_nPhonemes"+str(nPhonemes_train)+"-"+str(nPhonemes_eval)+ \
    "_IN"+str(input_noise)+"_WN"+str(weight_noise)+"_DR"+str(dropout)+ \
    "_ES"+".pt"


############## Data Loading ####################
with open(readname, 'rb') as f:
    data = pkl5.load(f)
x_train, y_train, x_val, y_val, x_test, y_test = data

if nPhonemes_read == 61 and nPhonemes_train == 39:
    y_train = mapfunc_val61_val39_list(y_train)
    y_val = mapfunc_val61_val39_list(y_val)
    y_test = mapfunc_val61_val39_list(y_test)


############## Data Processing ###################
# x padded is a list of Tensors with size (seq_len, DIM)
x_train_padded = paddingX(x_train, seq_len) 
x_val_padded = paddingX(x_val, seq_len)
x_test_padded = paddingX(x_test, seq_len)
# y padded is a list of Tensors with size (seq_len), padded mark is -1
y_train_padded = paddingy(y_train, seq_len) 
y_val_padded = paddingy(y_val, seq_len)
y_test_padded = paddingy(y_test, seq_len)
# turning x into a 3-D tensor, shape(N_training_data, seq_len, DIM)
x_train = torch.cat(x_train_padded, 0).reshape(-1,seq_len,DIM) # 
x_val = torch.cat(x_val_padded, 0).reshape(-1,seq_len,DIM)
x_test = torch.cat(x_test_padded, 0).reshape(-1,seq_len,DIM)
# turning y into a 2-D tensor, shape(N_training_data, seq_len)
y_train = torch.cat(y_train_padded, 0).reshape(-1,seq_len)
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


############### Model and Relative Functions Defining ###################
def mapfunc_val61_val39_tensor(y):
    '''
    input: 1D torch.Tensor 0-60 label sequences
    output: 1D torch.Tensor 0-38 label sequences
    ''' 
    y_mapped = []
    for val61 in y:
        val39 = val61_to_val39_dict[val61]
        y_mapped.append(val39)
    y_mapped = torch.tensor(y_mapped).reshape(-1, 1)
    return y_mapped

def train_one_epoch(m, optimizer, dataloader, device, nPhonemes_train, nPhonemes_eval):
    '''
    Function for training model one epoch
    input:
        m: model
        optimizer: Adam by default
        dataloader: data for training, train_loader
        device: 'cuda' or 'cpu'
        nPhonemes_train: number of phonemes for output of the model
        nPhonemes_eval: number of phonemes for evaluating model performance
    returns:
        train_epoch_loss, train_epoch_acc
    '''

    # set model to trainning mode
    m.train() 

    # initialization for calcualtion of acc and loss
    corrects = 0
    ttl = 0
    n_minibatch = 0 # number of mini-batches
    mb_loss = 0 # loss for a mini-batch

    for step, (x,y) in enumerate(dataloader):
        
        # put data in device
        x = x.to(device)
        y = y.to(device)

        # clean accumulated gradients
        optimizer.zero_grad()

        # model output (N, S, D)
        y_pred = m(x)

        # mask out the meaningless outputs by ruling out padding part
        mask = y.ge(0)
        mask_expanded = mask.unsqueeze(-1).expand(-1,-1,nPhonemes_train)
        y = torch.masked_select(y,mask)
        y_pred = torch.masked_select(y_pred,mask_expanded).reshape(-1,nPhonemes_train)

        # add softmax layer for prediction to transfer model output to logistics
        y_soft = F.softmax(y_pred, dim=1)

        ttl += len(y)
        # the dimension of y_pred can only be 61 when predict 61, while the dimension of y_pred_o can be 39 or 61 when predict 39
        if nPhonemes_train == 61 and nPhonemes_eval == 39:
            y_pred61 = torch.max(y_soft,1)[1].detach().cpu()
            y_pred39 = y_pred61.apply_(lambda x: val61_to_val39_dict[x])
            y39 = y.detach().cpu().apply_(lambda x: val61_to_val39_dict[x])
            # use self-defined functions instead to save time for moving data between cpu and gpu
            #y_pred39 = mapfunc_val61_val39_tensor(y_pred61)
            #y39 = mapfunc_val61_val39_tensor(y)
            corrects += (y_pred39 == y39).sum().item()
        else:
            corrects += (torch.max(y_soft,1)[1] == y ).detach().cpu().sum().item()  

        # calculate loss and update weights
        loss = nn.CrossEntropyLoss()(y_pred, y)
        mb_loss += loss.detach().cpu().item()
        n_minibatch += 1
        #if step%10 ==0:   # print training loss every 10 mini-batches
            #print(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()

    return mb_loss/n_minibatch, corrects/ttl


def evaluate(m, dataloader, device, nPhonemes_train, nPhonemes_eval):
    '''
    Function for evaluating model performance
    input:
        m: model
        dataloader: val_loader for validation, test_loader for testing
        device: 'cuda' or 'cpu'
        nPhonemes_train: number of phonemes for output of the model
        nPhonemes_eval: number of phonemes for evaluating model performance
    output:
        eval_loss, eval_acc (validation or testing)
    '''
    m.eval() # set model to evaluation mode
    corrects = 0
    ttl = 0
    n_minibatch = 0
    mb_loss = 0
    with torch.no_grad(): # stop updating weights
        for step, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            y_pred = m(x)
            mask = y.ge(0)
            mask_expanded = mask.unsqueeze(-1).expand(-1,-1,nPhonemes_train)
            y = torch.masked_select(y,mask)  # mask out the meaningless part of labels, y is a 1D Tensor of ground truth Phoneme IDs as a long sequence
            y_pred = torch.masked_select(y_pred,mask_expanded).reshape(-1,nPhonemes_train)

            # Add softmax layer for prediction to transfer model output to logistics
            y_soft = F.softmax(y_pred, dim=1)

            ttl += len(y)
            if nPhonemes_train == 61 and nPhonemes_eval == 39:
                y_pred61 = torch.max(y_soft,1)[1].detach().cpu()
                y_pred39 = y_pred61.apply_(lambda x: val61_to_val39_dict[x])
                y39 = y.detach().cpu().apply_(lambda x: val61_to_val39_dict[x])
                corrects += (y_pred39 == y39).sum().item()
            else:
                corrects += (torch.max(y_soft,1)[1] == y ).detach().cpu().sum().item()

            mb_loss += nn.CrossEntropyLoss()(y_pred, y).detach().cpu().item()
            n_minibatch += 1

    return mb_loss/n_minibatch, corrects/ttl

def train_model(m, train_loader, val_loader, opt, device, nPhonemes_train, nPhonemes_eval, n_epochs, early_stop=False, step_LR=False, save_model=False, savemodelname=None):
    '''
    Function for training and evaluating the network by repeatly calling train_one_epoch() and evaluate() iteratively
    input:
        m: model
    output:
        train_loss: a list of train_epoch_loss for length of trained epochs
        val_loss: a list of validation_loss for length of trained epochs
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
        train_loss_i, train_acc_i = train_one_epoch(m,opt,train_loader,device, nPhonemes_train, nPhonemes_eval)
        print("training loss:", train_loss_i)
        print("training acc:", train_acc_i)
        # Validation
        val_loss_i, val_acc_i = evaluate(m,val_loader,device, nPhonemes_train, nPhonemes_eval)
        print("validation loss:", val_loss_i)
        print("validation acc:", val_acc_i)
        train_loss.append(train_loss_i)
        val_loss.append(val_loss_i)

        # Adjust learning rate
        if step_LR:
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

class LSTMASR(nn.Module):

    def __init__(self, in_dim=39, out_dim=39, noise_x=0, noise_w=0, dropout=0):
        '''
            PARAMETERS:
            in_dim: network input dimension, (DIM =) 39 or 123
            out_dim: network output dimension, (nPhonemes_train) = 39 or 61
            noise_x: std of zero-mean Gaussian noise on X, only effective during trainingThe std of the gaussian noise added to the input x in training
            noise_w: The std of the gaussian noise added to the weights in training
            dropout: dropout layer probabilities of the LSTM layers
        '''
        super(LSTMASR, self).__init__()
        self.input_size = in_dim
        self.hidden_size = 250
        self.output_size = out_dim
        self.noise_x = noise_x
        self.noise_w = noise_w
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, bias=False, batch_first = True, bidirectional=True, dropout = dropout)
            ## bias = False following other's implementation, double-check
            ## batch_first = True, the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.out = nn.Linear(self.hidden_size*2, self.output_size) # add a fully connected layer
        #self.weight_init()
        
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
            # Adding noise to the model weights for regularization during training
            with torch.no_grad():
                self.lstm.weight_hh_l0 += torch.normal(0, self.noise_w, self.lstm.weight_hh_l0.size()).to(self.lstm.weight_hh_l0.device)
                self.lstm.weight_hh_l1 += torch.normal(0, self.noise_w, self.lstm.weight_hh_l1.size()).to(self.lstm.weight_hh_l1.device)
        lstm_out, _ = self.lstm(X) # lstm_out is the the output features from the last layer of the LSTM, for each token
        linear_out = self.out(lstm_out)
        return linear_out # need to apply a softmax to generate the probabilities


################### Model Training ###################
# Define model
m = LSTMASR(in_dim=DIM, out_dim=nPhonemes_train, noise_x=input_noise, noise_w=weight_noise, dropout=dropout)
m.to(device)
opt = torch.optim.Adam(m.parameters(), lr=0.001, betas=(.9,.99))
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

# Train the model and save the best model
train_loss, val_loss = train_model(m, train_loader, val_loader, opt, device, 
                                   nPhonemes_train, nPhonemes_eval, n_epochs, 
                                   early_stop, step_LR, save_model, savemodelname)

# Test the best model
print("++++++++++++++++Testing Start++++++++++++++++++")
m_best = LSTMASR(in_dim=DIM, out_dim=nPhonemes_train, noise_x=input_noise, noise_w=weight_noise, dropout=dropout)
m_best.load_state_dict(torch.load(savemodelname))
m_best.eval()
m_best.to(device)
_, test_acc = evaluate(m_best, test_loader, device, nPhonemes_train, nPhonemes_eval)
print("test acc:", test_acc)
