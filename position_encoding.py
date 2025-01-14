import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        '''
        d_model: dimention of the model, number of word embedding values per token
        max_len: the maximum number of tokens our simpleLLm can process -- input and output combined
        
        PE(pos,2i) = sin(pos/ 10000^(2i/model))
        PE(pos,2i+1) = cos(pos/ 10000^(2i/model))
        '''
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        # torch.arange() = to create a sequence of numbers
        # unsqueeze(1) = turns the sequence to a column matrix
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        # setting step=2 results in the same sequence numbers that we would get if we multiplied i by 2. So we save ourselves a little math!
        
        div_term = 1/torch.tensor(10000)**(embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        # To ensure that pe gets moved to a GPU if we use one!

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]