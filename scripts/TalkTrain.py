import re, math, time
import numpy as np

from nltk.corpus import wordnet

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

def num_batches(train):

    for i, b in enumerate(train):
        pass

    return i + 1

def nopeak_mask(size, opt):
    "Mask out subsequent positions. aka subsequent_mask"
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  torch.from_numpy(np_mask) == 0
    if opt.device == torch.device("cuda"):
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, opt):
    
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
        
    return src_mask, trg_mask

def string2tensor(string, inputfield, explain=False):
    '''
    input:
        string (str) input sentence
        inputfield a PyTorch torchtext.data.Field object
        explain, set this to True if you want to see how the sentence was split 
    output:
        sequence of tokens (torch tensor of integers) shape  
    '''
    sentence = inputfield.preprocess(string)
    if explain: print(sentence)
    integer_sequence = []
    for tok in sentence:
        if inputfield.vocab.stoi[tok] != 0:
            integer_sequence.append(inputfield.vocab.stoi[tok])
        else:
            integer_sequence.append(get_synonym(tok, inputfield))
    return torch.LongTensor([integer_sequence])

def get_synonym(word, field, explain=False):
    syns = wordnet.synsets(word)
    for s in syns:
        if explain: print('synonym:', s.name())
        for l in s.lemmas():
            if explain: print('-lemma:', l.name())
            if field.vocab.stoi[l.name()] != 0:
                if explain: print('found in vocab', l.name())
                return field.vocab.stoi[l.name()]
    return 0 # if we cannot find a synonym, return 0

def talk_to_chloe(input_str, model, opt, infield, outfield):
    '''
    input:
        input_str is a string, it is what you want to say to the dialogue model
        model is a Transformer model with encoder, decoder and a last layer linear transformation
        opt is an options object with the maximum length of the output sequence opt.max_len
        infield and outfield are the data.fields that store the vocabulary
    output:
        an output string response from the dialogue model
    
    Note: this version assumes we are evaluating the model on CPU 
    '''
    model.eval()
    model.cpu()
    input_sequence = string2tensor(input_str, infield , explain=True) # string to tensor 
    input_mask = (input_sequence != infield.vocab.stoi['<pad>']).unsqueeze(-2) #make input mask
    encoding = model.encoder(input_sequence, input_mask) # use the encoder rerepresent the input
    init_tok = outfield.vocab.stoi['<sos>'] # this is the integer for the start token
    decoder_input = torch.LongTensor([[init_tok]]) # use start token to initiate the decoder
    
    # continue obtaining the next decoder token until decoder outputs and end token or til max_len 
    for pos in range(opt.max_len):
        decoder_input_mask = nopeak_mask(size=pos+1, opt=opt) # make target mask, pos+1 casue pos starts at 0
        # the out vector contains the logits that are rebalanced by the softmax
        out = model.out(model.decoder(decoder_input, decoder_input_mask, encoding, input_mask))
        softout = F.softmax(out, dim=-1) 
        #softout is a categorical probability distribution over the output vocab
        distr = Categorical(probs=softout)
        action = distr.sample()[:,-1].unsqueeze(0) # sample from that distribution to get next token
        # concatenate that token to our running list of output tokens 
        decoder_input = torch.cat((decoder_input, action), dim=1) 
        # if the model outputs an end of sentence token, it is done with this sentence
        if outfield.vocab.itos[action] == '<eos>':
            # [0] because we are assuming batch size of 1 
            # [1:-1] excludes the start and end token from the output string 
            de_str = ' '.join([outfield.vocab.itos[tok] for tok in decoder_input[0][1:-1]])
            return de_str
        
    de_str = ' '.join([outfield.vocab.itos[tok] for tok in decoder_input[0]])
    return de_str

def linformer_trainer(model, train_data_iterator, train_options, test_data_iterator, test_options, optimizer, scheduler, scheduler_name):

    if torch.cuda.is_available() and train_options.device == torch.device("cuda"):
        print("==> a GPU was detected, model will be trained on GPU")
        model = model.cuda()
    else:
        print("==> training on cpu")

    model.train()
    start = time.time()
    best_loss = 100
    iters = len(train_data_iterator)
    for epoch in range(train_options.epochs):
        train_total_loss = 0
        for i, batch in enumerate(train_data_iterator):
            src = batch.listen
            trg = batch.reply
            diff = abs(len(batch.listen) - len(batch.reply))
            if len(batch.listen) > len(batch.reply):
                trg = torch.nn.functional.pad(input=trg, pad=(0,0,0,diff), mode='constant', value=0)
            elif len(batch.listen) < len(batch.reply):
                src = torch.nn.functional.pad(input=src, pad=(0,0,0,diff), mode='constant', value=0)
            # print("src shape", src.shape)
            diff = abs(train_options.batchsize - src.shape[1])
            if train_options.batchsize > src.shape[1]:
                src = torch.nn.functional.pad(input=src, pad=(0,diff,0,0), mode='constant', value=0)
            if train_options.batchsize > trg.shape[1]:
                trg = torch.nn.functional.pad(input=trg, pad=(0,diff,0,0), mode='constant', value=0)
            # print("src shape", src.shape)
            preds = model(src,trg)
            ys = trg.contiguous().view(-1)
            preds = preds.contiguous().view(-1, preds.size(-1))
            optimizer.zero_grad()
            train_batch_loss = F.cross_entropy(preds, ys, ignore_index = train_options.trg_pad)
            train_batch_loss.backward()
            optimizer.step()
            train_total_loss += train_batch_loss.item()
            if scheduler_name == "cosine":
                scheduler.step(epoch + i / iters)
            if scheduler_name == "warmup": 
                scheduler.step()
            # print("batch loss", train_batch_loss)
        if scheduler_name == "warmup":
            scheduler.print_lr(epoch+i)
        train_epoch_loss = train_total_loss/(num_batches(train_data_iterator)+1)

        if scheduler_name == "plateau": 
            scheduler.step(train_epoch_loss) 

        model.eval()
        test_total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(test_data_iterator): 
                src = batch.listen
                trg = batch.reply
                diff = abs(len(batch.listen) - len(batch.reply))
                if len(batch.listen) > len(batch.reply):
                    trg = torch.nn.functional.pad(input=trg, pad=(0,0,0,diff), mode='constant', value=0)
                elif len(batch.listen) < len(batch.reply):
                    src = torch.nn.functional.pad(input=src, pad=(0,0,0,diff), mode='constant', value=0)
                # print("src shape", src.shape) 
                diff = abs(test_options.batchsize - src.shape[1])
                if test_options.batchsize > src.shape[1]:
                    src = torch.nn.functional.pad(input=src, pad=(0,diff,0,0), mode='constant', value=0)
                if test_options.batchsize > trg.shape[1]:
                    trg = torch.nn.functional.pad(input=trg, pad=(0,diff,0,0), mode='constant', value=0)
                # print("src shape", src.shape)              
                preds = model(src,trg)
                ys = trg.contiguous().view(-1)
                preds = preds.contiguous().view(-1, preds.size(-1))
                test_batch_loss = F.cross_entropy(preds, ys, ignore_index = test_options.trg_pad)
                test_total_loss += test_batch_loss.item()

            test_epoch_loss = test_total_loss/(num_batches(test_data_iterator)+1)

        # if scheduler_name == "plateau": 
        #     scheduler.step(test_epoch_loss) 

        model.train()

        # scheduler.step()

        if train_epoch_loss < best_loss:
            best_loss = train_epoch_loss
            torch.save(model.state_dict(), train_options.save_path)
        # if test_epoch_loss < best_loss:
        #     best_loss = test_epoch_loss
        #     torch.save(model.state_dict(), train_options.save_path)
        print("%.3fm: train epoch *%d*, loss = *%.3f*" %((time.time() - start)//60, epoch+1, train_epoch_loss), end=", ")
        print("%.3fm: test epoch *%d*, loss = *%.3f*, best loss = *%.3f*" %((time.time() - start)//60, epoch+1, test_epoch_loss, best_loss) , flush=True)
        train_total_loss = 0
        test_total_loss = 0

def transformer_trainer(model, train_data_iterator, train_options, test_data_iterator, test_options, optimizer, scheduler, scheduler_name):

    if torch.cuda.is_available() and train_options.device == torch.device("cuda"):
        print("==> a GPU was detected, model will be trained on GPU")
        model = model.cuda()
    else:
        print("==> training on cpu")

    model.train()
    start = time.time()
    best_loss = 100
    
    iters = len(train_data_iterator)
    for epoch in range(train_options.epochs):
        train_total_loss = 0
        for i, batch in enumerate(train_data_iterator): 
            src = batch.listen.transpose(0,1)
            trg = batch.reply.transpose(0,1)
            # print("src", src.shape)
            # print("trg", trg.shape)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, train_options)
            preds = model(src, src_mask, trg_input, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            train_batch_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), 
                                         ys, ignore_index = train_options.trg_pad)
            train_batch_loss.backward()
            optimizer.step()
            train_total_loss += train_batch_loss.item()

            if scheduler_name == "cosine": 
                scheduler.step(epoch + i / iters)
            if scheduler_name == "warmup": 
                scheduler.step()
            # print("batch loss", train_batch_loss)
        if scheduler_name == "warmup":
            scheduler.print_lr(epoch+i)
        train_epoch_loss = train_total_loss/(num_batches(train_data_iterator)+1)

        if scheduler_name == "plateau": 
            scheduler.step(train_epoch_loss)

        model.eval()
        test_total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(test_data_iterator): 
                src = batch.listen.transpose(0,1)
                trg = batch.reply.transpose(0,1)
                trg_input = trg[:, :-1]
                src_mask, trg_mask = create_masks(src, trg_input, test_options)
                preds = model(src, src_mask, trg_input, trg_mask)
                ys = trg[:, 1:].contiguous().view(-1)
                test_batch_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), 
                                             ys, ignore_index = test_options.trg_pad)
                test_total_loss += test_batch_loss.item()

            test_epoch_loss = test_total_loss/(num_batches(test_data_iterator)+1)

        # if scheduler_name == "plateau": 
        #     scheduler.step(test_epoch_loss)
                    
        model.train()
        

        if train_epoch_loss < best_loss:
            best_loss = train_epoch_loss
            torch.save(model.state_dict(), train_options.save_path)
        # if test_epoch_loss < best_loss:
        #     best_loss = test_epoch_loss
        #     torch.save(model.state_dict(), train_options.save_path)
        print("%.3fm: train epoch *%d*, loss = *%.3f*" %((time.time() - start)//60, epoch+1, train_epoch_loss), end=", ")
        print("%.3fm: test epoch *%d*, loss = *%.3f*, best loss = *%.3f*" %((time.time() - start)//60, epoch+1, test_epoch_loss, best_loss) , flush=True)
        train_total_loss = 0
        test_total_loss = 0

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts. 
    #scheduler = CosineWithRestarts(optimizer, T_max=num_batches(data_iter))
    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.,
                 verbose=False) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.0
        self._updated_cycle_len = T_max
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {}.'.format(group, lr))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: adjusting learning rate'
                      ' of group {} to {}.'.format(epoch_str, group, lr))

class AttentionLRscheduler:
    def __init__(self, model_size, warmup_steps, optimizer, verbose=False):
        
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
        self.verbose = verbose
        
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
        
    def step(self):
        # Increment the number of steps each time we call the step function
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # update the learning rate
        self.lr = lr
        self.optimizer.step() 
        if self.verbose==True:
            print(f"Adjusting learning rate to {self.lr}.")

class AdamWarmup:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup_steps, optimizer, verbose=False):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self._rate = 0
        self.verbose = verbose
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))) 

    def print_lr(self, epoch=None):
        """Display the current learning rate.
        """
        if self.verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' to {}.'.format(self._rate))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: adjusting learning rate'
                      ' to {}.'.format(epoch_str, self._rate))

        