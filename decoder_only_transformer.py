import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from position_encoding import PositionEncoding
from attention import Attention

class DecoderOnlyTransformer(L.LightningModule):
    '''
    Doing it this way, rather than having every class inherit from
    LightningModule, allows us to take advantage of everything Lightning offers without the overhead of inheriting it multiple times.
    '''
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        # max_len: Maximum lenght of the input + output
        super().__init__()

        self.we = nn.Embedding(num_embeddings= num_tokens, embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
        # We create the mask that will prevent early tokens from looking at late tokens when we calculate Attention.
        mask=mask==0

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=mask)
        
        residual_connection_values = position_encoded + self_attention_values
        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output