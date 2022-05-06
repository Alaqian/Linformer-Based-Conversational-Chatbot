import math, copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scripts.TransLinsUtils import *

def get_EF(input_size, dim, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin

class LinformerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, linear_dim, dim_k = None, dropout = 0.1):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.linear_dim = linear_dim
        self.dim_k = dim_k if dim_k else emb_dim // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(emb_dim,self.dim_k*num_heads)
        self.k_linear = nn.Linear(emb_dim,self.dim_k*num_heads)
        self.v_linear = nn.Linear(emb_dim,self.dim_k*num_heads)
        self.e_linear = nn.Linear(emb_dim,self.linear_dim*num_heads)
        self.f_linear = nn.Linear(emb_dim,self.linear_dim*num_heads)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.dim_k*num_heads,emb_dim)

    def attention(self, q, k, v, dim_k, mask=None, dropout=None, explain=False):
        k = k.transpose(-2, -1)
        if explain: print('q, k', q.shape, k.shape)
        # matrix multiplication is done using the last two dimensions
        # (batch_size,num_heads,q_seq_len,dim_k)X(batch_size,num_heads,dim_k,k_seq_len)
        #(batch_size,num_heads,q_seq_len,k_seq_len)
        scores = torch.matmul(q, k) / math.sqrt(dim_k) 
        if explain: print('scores.shape', scores.shape)
        if mask is not None:
            mask = mask.unsqueeze(1)
            if explain: print('mask.shape', mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9) 
        softscores = F.softmax(scores, dim=-1)
        if dropout is not None: softscores = dropout(softscores)
            
        #(batch_size,num_heads,seq_len,seq_len)X(batch_size,num_heads,seq_len,dim_k)
        output = torch.matmul(softscores, v)
        return output, scores #=(batch_size,num_heads,seq_len,dim_k)

    def linearAttention(self, q, k, v, e, f, dim_k, mask=None, dropout=None, explain=False):
        e = e.transpose(-2, -1)
        if explain: print('e, k before matmul: ', e.shape, k.shape)
        # Linear attention
        k = torch.matmul(e, k)
        if explain:  print("e*k", k.shape)
        k = k.transpose(-2, -1)
        if explain: print('q, k before matmul: ', q.shape, k.shape)
        if explain:   print("q, k", torch.matmul(q, k).shape)
        # matrix multiplication is done using the last two dimensions
        # (batch_size,num_heads,q_seq_len,dim_k)X(batch_size,num_heads,dim_k,k_seq_len)
        # (batch_size,num_heads,q_seq_len,k_seq_len)
        scores = torch.matmul(q, k) / math.sqrt(dim_k) 
        # scores = scores.transpose(-2, -1)
        if explain: print('scores.shape', scores.shape)
        # print("score", scores)
        if mask is not None:
            if explain: print('mask.shape before narrow', mask.shape)
            mask = mask.narrow(2,0,scores.shape[-1])
            if explain: print('mask.shape before unsqueeze', mask.shape)
            # print("mask before unsqueeze", mask)
            mask = mask.unsqueeze(1)
            # print("mask", mask)
            if explain: print('mask.shape after unsqueeze', mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9) 
        softscores = F.softmax(scores, dim=-1)
        if explain: print('softscores.shape', softscores.shape)
        # if explain: print('softscores',softscores)
        # softscores = softscores.transpose(-2, -1)
        if dropout is not None: softscores = dropout(softscores)
        f = f.transpose(-2, -1)
        if explain: print('f, v', f.shape, v.shape)
        v = torch.matmul(f, v)
        #(batch_size,num_heads,seq_len,seq_len)X(batch_size,num_heads,seq_len,dim_k)
        output = torch.matmul(softscores, v)
        if explain: print('output.shape', output.shape)
        # if explain: print('output', output)
        return output, scores #=(batch_size,num_heads,seq_len,dim_k)
    
    def forward(self, q, k, v, mask=None, explain=False):
        '''
        inputs:
            q has shape (batch size, q_sequence length, embedding dimensions)
            k,v are shape (batch size, kv_sequence length, embedding dimensions)
            source_mask of shape (batch size, 1, kv_sequence length)
        outputs: sequence of vectors, re-represented using attention
            shape (batch size, q_sequence length, embedding dimensions)
        use:
            The encoder layer places the same source vector sequence into q,k,v 
            and source_mask into mask.
            The decoder layer uses this twice, once with decoder inputs as q,k,v 
            and target mask as mask. then with decoder inputs as q, encoder outputs
            as k, v and source mask as mask
        '''
        # k,q,v are each shape (batch size, sequence length, dim_k * num_heads)
        batch_size = q.size(0)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        e = self.e_linear(k)
        f = self.f_linear(v)
        if explain: print("(batch size, sequence length, dim_k * num_heads)", k.shape)
        # k,q,v are each shape (batch size, sequence length, num_heads, dim_k)
        k = k.view(batch_size,-1,self.num_heads,self.dim_k)
        q = q.view(batch_size,-1,self.num_heads,self.dim_k)
        v = v.view(batch_size,-1,self.num_heads,self.dim_k)
        e = e.view(batch_size,-1,self.num_heads,self.linear_dim)
        f = f.view(batch_size,-1,self.num_heads,self.linear_dim)
        # transpose to shape (batch_size, num_heads, sequence length, dim_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        e = e.transpose(1,2)
        f = f.transpose(1,2)
        if explain: print("(batch_size,num_heads,seq_length,dim_k)",k.shape)
        # calculate attention using function we will define next
        if explain:  print('length of e, k', e.shape[-1], k.shape[-2])
        if(e.shape[-1] < k.shape[-2]):
            if explain:  print('go linear')
            attn, scores = self.linearAttention(q, k, v, e, f, self.dim_k, mask, self.dropout, explain)
        else:
            if explain:  print('go basic')
            attn, scores = self.attention(q, k, v, self.dim_k, mask, self.dropout, explain)
        
        if explain: print("attn(batch_size,num_heads,seq_length,dim_k)", attn.shape)
        # concatenate heads and 
        concat=attn.transpose(1,2).contiguous().view(batch_size,-1,self.dim_k*self.num_heads)
        if explain: print("concat.shape", concat.shape)
        # put through final linear layer
        output = self.out(concat)
        if explain: print("LinformerMultiHeadAttention output.shape", output.shape)
        return output, scores

class LinformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, linear_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.attn = LinformerMultiHeadAttention(heads, emb_dim, linear_dim, dropout=dropout)
        self.norm_2 = Norm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout=dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, vector_sequence, mask):
        '''
        input:
            vector_sequence of shape (batch size, sequence length, embedding dimensions)
            source_mask (mask over input sequence) of shape (batch size, 1, sequence length)
        output: sequence of vectors after embedding, postional encoding, attention and normalization
            shape (batch size, sequence length, embedding dimensions)
        '''
        x2 = self.norm_1(vector_sequence)
        x2_attn, x2_scores = self.attn(x2,x2,x2,mask)
        vector_sequence = vector_sequence + self.dropout_1(x2_attn)
        x2 = self.norm_2(vector_sequence)
        vector_sequence = vector_sequence + self.dropout_2(self.ff(x2))
        return vector_sequence

class LinformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, linear_dim, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim, dropout=dropout)
        self.layers = get_clones(LinformerEncoderLayer(emb_dim, linear_dim, heads, dropout), n_layers)
        self.norm = Norm(emb_dim)
    def forward(self, source_sequence, source_mask):
        '''
        input:
            source_sequence (sequence of source tokens) of shape (batch size, sequence length)
            source_mask (mask over input sequence) of shape (batch size, 1, sequence length)
        output: sequence of vectors after embedding, postional encoding, attention and normalization
            shape (batch size, sequence length, embedding dimensions)
        '''
        vector_sequence = self.embed(source_sequence)
        vector_sequence = self.pe(vector_sequence)
        for i in range(self.n_layers):
            vector_sequence = self.layers[i](vector_sequence, source_mask)
        vector_sequence = self.norm(vector_sequence)
        return vector_sequence

class LinformerDecoderLayer(nn.Module):

    def __init__(self, emb_dim, linear_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)
        self.norm_3 = Norm(emb_dim)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = LinformerMultiHeadAttention(heads, emb_dim, linear_dim, dropout=dropout)
        self.attn_2 = LinformerMultiHeadAttention(heads, emb_dim, linear_dim, dropout=dropout)
        self.ff = FeedForward(emb_dim, dropout=dropout)

    def forward(self, de_out, de_mask, en_out, en_mask):
        '''
        inputs:
            de_out - decoder ouputs so far (batch size, output sequence length, embedding dimensions)
            de_mask (batch size, output sequence length, output sequence length)
            en_out - encoder output (batch size, input sequence length, embedding dimensions)
            en_mask (batch size, 1, input sequence length)
        ouputs:
            de_out (next decoder output) (batch size, output sequence length, embedding dimensions)
        '''
        de_nrm = self.norm_1(de_out)
        #Self Attention 
        self_attn, self_scores = self.attn_1(de_nrm, de_nrm, de_nrm, de_mask)
        de_out = de_out + self.dropout_1(self_attn)
        de_nrm = self.norm_2(de_out)
        #DecoderEncoder Attention
        en_attn, en_scores = self.attn_2(de_nrm, en_out, en_out, en_mask) 
        de_out = de_out + self.dropout_2(en_attn)
        de_nrm = self.norm_3(de_out)
        de_out = de_out + self.dropout_3(self.ff(de_nrm))
        return de_out

class LinformerDecoder(nn.Module):
    '''
    If your target sequence is `see` `ya` and you want to train on the entire 
    sequence against the target, you would use `<sos>` `see`  `ya`
    as the de_out (decoder ouputs so far) and compare the 
    output de_out (next decoder output) `see` `ya` `<eos>` 
    as the target in the loss function. The inclusion of the `<sos>`
    for the (decoder ouputs so far) and `<eos>` for the 
    '''
    def __init__(self, vocab_size, emb_dim, linear_dim, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim, dropout=dropout)
        self.layers = get_clones(LinformerDecoderLayer(emb_dim, linear_dim, heads, dropout), n_layers)
        self.norm = Norm(emb_dim)
    def forward(self, de_toks, de_mask, en_vecs, en_mask):
        '''
        inputs:
            de_toks - decoder ouputs so far (batch size, output sequence length)
            de_mask (batch size, output sequence length, output sequence length)
            en_vecs - encoder output (batch size, input sequence length, embedding dimensions)
            en_mask (batch size, 1, input sequence length)
        outputs:
            de_vecs - next decoder output (batch size, output sequence length, embedding dimensions)

        '''
        x = self.embed(de_toks)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, de_mask, en_vecs, en_mask)
        return self.norm(x)

class Linformer(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, emb_dim, linear_dim, n_layers, heads, dropout):
        super().__init__()
        self.encoder = LinformerEncoder(in_vocab_size, emb_dim, linear_dim, n_layers, heads, dropout)
        self.decoder = LinformerDecoder(out_vocab_size, emb_dim, linear_dim, n_layers, heads, dropout)
        self.out = nn.Linear(emb_dim, out_vocab_size)
    def forward(self, src_seq, src_mask, trg_seq,  trg_mask):
        e_output = self.encoder(src_seq, src_mask)
        d_output = self.decoder(trg_seq, trg_mask, e_output, src_mask)
        output = self.out(d_output)
        return output

