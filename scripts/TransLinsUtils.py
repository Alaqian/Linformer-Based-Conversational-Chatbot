import math, copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):

    def __init__(self, emb_dim, max_seq_len = 1000, dropout = 0.1):
        super().__init__()
        self. emb_dim =  emb_dim
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on pos and i
        # this matrix of shape (1, input_seq_len, emb_dim) is
        # cut down to shape (1, input_seq_len, emb_dim) in the forward pass
        # to be broadcasted across each sample in the batch 
        pe = torch.zeros(max_seq_len, emb_dim)
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                wavelength = 10000 ** ((2 * i)/ emb_dim)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0) # add a batch dimention to your pe matrix 
        self.register_buffer('pe', pe)#block of data(persistent buffer)->temporary memory
 
    def forward(self, x):
        '''
        input: sequence of vectors  
               shape (batch size, input sequence length, vector dimentions)
        output: sequence of vectors of same shape as input with positional
                aka time encoding added to each sample 
                shape (batch size, input sequence length, vector dimentions)
        '''
        x = x * math.sqrt(self. emb_dim) # make embeddings relatively larger
        # print(x.shape, x.size(1))
        seq_len = x.size(1)
        # print(seq_len)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe #add constant to embedding
        return self.dropout(x)
    
def get_clones(module, N):
    '''
    example usage:
        
        # initialize n_layers deep copies of the same encoder
        self.layers = get_clones(EncoderLayer(emb_dim, heads, dropout), n_layers)
        
        # usage, apply n_layers transformations to x
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)  
    '''
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Norm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-6):
        super().__init__()
        self.size = emb_dim
        # alpha and bias are learnable parameters that scale and shift
        # the representations respectively, aka stretch and translate 
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps #prevents divide by zero explosions 
    
    def forward(self, x):
        '''
        input/output: x/norm shape (batch size, sequence length, embedding dimensions)
        '''
        norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        norm = self.alpha * norm + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(emb_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, emb_dim)
    
    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

