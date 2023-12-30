#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|    
#        Class_Dataset                        

import torch.utils.data as data
import torch
import random
import collections

class Vocab: 
    """
    input:              corpus_str
    getitem:            token to index
    vocab.idx_to_token: a list to transform index to token
    """
    def __init__(self, corpus:list[str], min_freq=0, reserved_tokens=None):
        """_summary_

        Args:
            corpus (list[str]): a 1-d list that has str
            min_freq (int, optional): min_frequence. Defaults to 0.
            reserved_tokens (_type_, optional): reserved_tokens. Defaults to None.
        """
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = self.count_corpus(corpus)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):#token to index
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):#index to token
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    def count_corpus(self,corpus):  #@save
        """统计词元的频率"""
        return collections.Counter(corpus)


class my_data(data.Dataset):
    def __init__(self,corpus_str,num_steps) -> None:
        super().__init__()
        self.num_steps=num_steps
        self.__vocab=Vocab(corpus_str)
        self.corpus_num=torch.tensor([self.__vocab[token] for token in corpus_str])
        self.corpus_num=self.corpus_num[(random.randint(0,num_steps-1)):]
    @property
    def vocab(self):
        return self.__vocab
    def __len__(self):
        return (len(self.corpus_num)-1)//self.num_steps
    def __getitem__(self, index):
        start=index*self.num_steps
        end=start+self.num_steps
        return (self.corpus_num[start:end],self.corpus_num[start+1:end+1])































