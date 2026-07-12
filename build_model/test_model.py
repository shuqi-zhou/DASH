import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF
import torch.nn.functional as F
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.layers import GlobalPointer

class PointerNet(BaseModel):
    def __init__(self, input_size, hidden_size, output_dim, num_layers, dropout, bidirectional):
        super(PointerNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional = bidirectional)
        d = 2 if bidirectional else 1
        # self.fc = nn.Linear(d*hidden_size, output_dim)
        self.global_pointer = GlobalPointer(hidden_size=d*hidden_size, heads=output_dim, head_size=d*hidden_size)
        
    def forward(self, packed_sequences):
        packed_output, (hn, cn) = self.lstm(packed_sequences)
        pad_mask = packed_sequences[:,:,0].ne(0).long()
        logit = self.global_pointer(packed_output, pad_mask)
        return logit

