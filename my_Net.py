#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|    
#        Class_Net         
import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
import my_Dataset
def parse_Rnn_layers(layers):
    mid=layers[0]
    dic=defaultdict(int)
    label=0
    for i in layers:
        if i==mid:
            dic[str(i)+f'_{label}']+=1
            continue
    else:
        mid=i
        label+=1
        dic[str(i)+f'_{label}']+=1

    return dic

class Net(nn.Module):
    def __init__(self,Rnn_type:str,Rnn_layers:tuple,Linear_layers:tuple,
                 vocab:my_Dataset.Vocab,device):
        super().__init__()
        self.Rnn_type=Rnn_type
        self.device=device
        self.vocab=vocab
        self.feature_dims=Linear_layers[-1]
        dic=parse_Rnn_layers(Rnn_layers)
        RNN_list=[]
        if Rnn_type=='RNN':
            for index,items in enumerate(dic.items()):
                if index==0:
                    RNN_list.append(nn.RNN(self.feature_dims,previous_rnn:=int(items[0].split('_')[0]),items[1]))
                else:
                    RNN_list.append(nn.RNN(previous_rnn,previous_rnn:=int(items[0].split('_')[0]),items[1]))
        elif Rnn_type=='GRU':
            for index,items in enumerate(dic.items()):
                if index==0:
                    RNN_list.append(nn.GRU(self.feature_dims,previous_rnn:=int(items[0].split('_')[0]),items[1]))
                else:
                    RNN_list.append(nn.GRU(previous_rnn,previous_rnn:=int(items[0].split('_')[0]),items[1]))
        elif Rnn_type=='LSTM':
            for index,items in enumerate(dic.items()):
                if index==0:
                    RNN_list.append(nn.LSTM(self.feature_dims,previous_rnn:=int(items[0].split('_')[0]),items[1]))
                else:
                    RNN_list.append(nn.LSTM(previous_rnn,previous_rnn:=int(items[0].split('_')[0]),items[1]))
        else:
            raise ValueError('Rnn_type only accept "RNN" or "GRU" or "LSTM"' )
        self.RNN_Seq=nn.Sequential(*RNN_list)
        Dense_list=[]
        for index,layer in enumerate(Linear_layers):
            if index==0:
                Dense_list.append(nn.Linear(previous_rnn,previous_dense:=layer))
            else:
                Dense_list.append(nn.Linear(previous_dense,previous_dense:=layer))
        self.Dense_Seq=nn.Sequential(*Dense_list)
    
    def init_state(self,batch_size):
        states_list=[]
        if self.Rnn_type!='LSTM':
            for seq in self.RNN_Seq:
                states_list.append(torch.zeros(seq.num_layers,batch_size,seq.hidden_size,device=self.device))
        else:
            for seq in self.RNN_Seq:
                states_list.append((torch.zeros(seq.num_layers,batch_size,seq.hidden_size,device=self.device),
                                   torch.zeros(seq.num_layers,batch_size,seq.hidden_size,device=self.device)))
        return states_list
    def forward(self,X,states_list=None):
        if states_list==None:
            states_list=self.init_state(X.shape[0])
        
        X=F.one_hot(X.T.long(),self.feature_dims)
        X=X.float()
        for index,seq in enumerate(self.RNN_Seq):
            X,states_list[index]=seq(X,states_list[index])
        X=X.reshape(-1,X.shape[-1])#它的输出形状是(时间步数*批量大小,num_hidden)。
        return self.Dense_Seq(X),states_list #它的输出形状是(时间步数*批量大小,词表大小)。
    
    def predict(self,prefix,num):
        states_list=self.init_state(batch_size=1)
        outputs=[]
        outputs.append(self.vocab[prefix[0]])
        get_input=lambda :torch.tensor(outputs[-1],device=self.device).reshape(1,1)
        for token in prefix[1:]:
            _,states_list=self.forward(get_input(),states_list)#warm up
            outputs.append(self.vocab[token])
        for _ in range(num):
            y,states_list=self.forward(get_input(),states_list)
            outputs.append(int(y.argmax(dim=1).reshape(-1)))
        return ''.join([self.vocab.idx_to_token[i] for i in outputs])















