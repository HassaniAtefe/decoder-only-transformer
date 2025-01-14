import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

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

        ## We are set the seed so that you can get the same results as me.
        L.seed_everything(seed=42)
        
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.we = nn.Embedding(num_embeddings= num_tokens, embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)), device=self.device))
        # We create the mask that will prevent early tokens from looking at late tokens when we calculate Attention.
        mask=mask==0

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=mask)
        
        residual_connection_values = position_encoded + self_attention_values
        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output
    
    def configure_optimizers(self):
        # Configure_optimizers() simply passes the parameters we want to optimize to the optimzes and sets the learning rate
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_idx):
        # training_step() is called by Lightning trainer when we want to train the model.
        input_tokens, labels = batch # collect input
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0]) 
        return loss