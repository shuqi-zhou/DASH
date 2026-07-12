import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, packed_sequences):

        packed_output, (hn, cn) = self.lstm(packed_sequences)
        output_padded, output_lengths = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        output_unpacked = self.fc(output_padded)

        return output_unpacked


class LSTM_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers, dropout, bidirectional):
        super(LSTM_CRF, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional = bidirectional)
        d = 2 if bidirectional else 1
        self.fc = nn.Linear(d*hidden_size, output_dim)
        self.crf = CRF(output_dim, batch_first = True)
        
    def forward(self, packed_sequences, loss_mask=None, pad_mask=None, labels=None):
        packed_output, (hn, cn) = self.lstm(packed_sequences)
        output_padded, output_lengths = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        emission= self.fc(output_padded)
        emission_log_probs = F.log_softmax(emission, dim=2)
        if labels is not None:    
            loss = - self.crf(emission_log_probs, labels, mask=loss_mask, reduction='token_mean')

            prediction = self.crf.decode(emission_log_probs, mask=pad_mask)
            return loss, prediction
        else:         
            prediction = self.crf.decode(emission_log_probs, mask=pad_mask)
            return prediction
        

# packed_sequences = sequences  
# mask = label_mask 
# packed_output, (hn, cn) = model.lstm(packed_sequences)
# output_padded, output_lengths = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

# torch.isinf(output_padded).any()
# output_unpacked = model.fc(output_padded)
# emission= model.fc(output_padded)
# emission_log_probs = F.log_softmax(emission, dim=2)
# prediction = model.crf._viterbi_decode(emission_log_probs, mask=mask)
# change prediction as torch tensor