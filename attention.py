import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model=2):
        # d_model: number of word embedding values per token
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        # In the original Transformers manuscript, they don't add additional bias terms when calculating Attention, so we won't either by setting bias = False.
        # As a result, we end up with an object we're calling W_q with the currently untrained Weights needed to calculate Query values.
       
        self.row_dim = 0
        self.col_dim = 1
        # Just to give us flexibility to input training data in sequentially or in batches, we creat some variables to keep track of which indices are for rows and columns.

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        ''' We are doing for the sake of flexibility is allowing the Query, Key and Values 
            to be calculated from different token encodings.
        '''
        
        q = self.W_q(encodings_for_q)
        k = self.W_q(encodings_for_k)
        v = self.W_q(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
            # Masking is used to prevent early tokens from cheating and looking at later tokens.
        
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
