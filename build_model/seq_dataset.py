import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import numpy as np

# Dummy implementation to replace bert4torch.snippets.sequence_padding
# The original function is not actually used in this codebase (commented out)
def sequence_padding(sequences, padding_value=0, seq_dims=1):
    """Placeholder function - not actually used in this code."""
    return sequences

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        Args:
        sequences (list of arrays): A list where each element is an array of shape (n_i, 1).
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label, weight = self.sequences[index]
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(-1) if sequence.ndim == 1 else torch.tensor(sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        if isinstance(weight, (str, bytes)) or not hasattr(weight, '__len__'):
            weight = np.ones(len(label), dtype=np.float32)
        weight = torch.tensor(weight, dtype=torch.float32)
        return (x, label, weight)
    
# def collate_fn(batch):
#     """
#     A custom collate function to pad the sequences to the same length in each batch.
#     """
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     sequences, labels, weights = zip(*batch)
#     lengths = [len(seq) for seq in sequences]

#     # Pad sequences
    
#     labels_padded = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-1)
#     weights_padded = rnn_utils.pad_sequence(weights, batch_first=True, padding_value=0)
#     sequences_padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=0)
    
#     packed_sequences = rnn_utils.pack_padded_sequence(sequences_padded, lengths, batch_first=True)
    
#     return packed_sequences, labels_padded, weights_padded


def collate_fn(batch):
    """
    Optimized collate function using pre-allocated arrays and numpy operations
    """
    # Sort by length first
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels, weights = zip(*batch)
    
    # Get max length and batch size
    max_len = len(sequences[0])
    batch_size = len(sequences)
    seq_dim = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
    
    # Pre-allocate numpy arrays
    sequences_array = np.zeros((batch_size, max_len, seq_dim), dtype=np.float32)
    labels_array = np.full((batch_size, max_len), -1, dtype=np.int64)
    weights_array = np.zeros((batch_size, max_len), dtype=np.float32)
    lengths = np.array([len(seq) for seq in sequences], dtype=np.int64)
    
    # Fill arrays
    for i, (seq, label, weight) in enumerate(zip(sequences, labels, weights)):
        curr_len = lengths[i]
        if seq_dim == 1:
            sequences_array[i, :curr_len, 0] = seq
        else:
            sequences_array[i, :curr_len] = seq
        labels_array[i, :curr_len] = label
        weights_array[i, :curr_len] = weight
    
    # Convert to torch tensors
    sequences_tensor = torch.from_numpy(sequences_array)
    labels_tensor = torch.from_numpy(labels_array)
    weights_tensor = torch.from_numpy(weights_array)
    
    # Pack sequences
    packed_sequences = rnn_utils.pack_padded_sequence(
        sequences_tensor, 
        lengths, 
        batch_first=True
    )
    
    return packed_sequences, labels_tensor, weights_tensor


class BucketBatchSampler:
    def __init__(self, data_source, batch_size, bucket_boundaries=[400, 800, 1200, 1600, 1800]):
        self.data_source = data_source
        self.batch_size = batch_size
        self.boundaries = sorted(bucket_boundaries)
        
        # Group samples by length into buckets
        self.buckets = {length: [] for length in self.boundaries}
        for idx in range(len(data_source)):
            sequence = data_source[idx][0]  
            seq_length = len(sequence)

            for boundary in self.boundaries:
                if seq_length <= boundary:
                    self.buckets[boundary].append(idx)
                    break
        
        self.buckets = {k: v for k, v in self.buckets.items() if len(v) > 0}
        
    def __iter__(self):
        # Shuffle indices within each bucket
        for bucket in self.buckets.values():
            np.random.shuffle(bucket)
            
        for bucket in self.buckets.values():
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]
    
    def __len__(self):
        return sum(len(bucket) // self.batch_size + (1 if len(bucket) % self.batch_size else 0)
                  for bucket in self.buckets.values())

def collate_triplet(batch):
    """
    Optimized collate function using pre-allocated tensors and numpy operations
    """
    # Sort by length first
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = len(batch[0][0])
    batch_size = len(batch)
    
    # Pre-allocate tensors
    sequences = np.zeros((batch_size, max_len, batch[0][0].shape[1]), dtype=np.float32)
    labels = np.zeros((batch_size, 2, max_len, max_len), dtype=np.int64)
    
    # Fill tensors
    for i, (sequence, seq_label, _) in enumerate(batch):
        curr_len = len(sequence)
        sequences[i, :curr_len] = sequence
        
        if len(seq_label) > 0:
            for (start, end), label in seq_label:
                label -= 1  # input label1, label2 mapping to 0, 1
                labels[i, label, start, end] = 1
    
    # Convert to torch tensors directly
    sequences = torch.from_numpy(sequences)
    labels = torch.from_numpy(labels)
    
    return sequences, labels

# def collate_triplet(batch):
#     """
#     A custom collate function to turn the label into shape (batch, class_num, seq_len, seq_len) pad the sequences to the same length in each batch.
#     """
     
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     batch_token_ids, batch_labels = [], []
#     lengths = []
#     for (sequence, seq_label, _) in batch:
#         lengths.append(len(sequence))
#         labels = np.zeros((2, len(sequence), len(sequence)))
#         if len(seq_label) > 0:
#             for (start, end), label in seq_label:
#                 label -= 1 #input label1, label2 mapping to 0, 1
#                 labels[label, start, end] = 1

#         batch_token_ids.append(sequence) 
#         batch_labels.append(labels)
    
#     # sequences_padded = rnn_utils.pad_sequence(batch_token_ids, batch_first=True, padding_value=0)
#     # packed_sequences = rnn_utils.pack_padded_sequence(sequences_padded, lengths, batch_first=True)
#     packed_sequences = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.float32)
#     batch_labels = torch.tensor(sequence_padding(batch_labels, seq_dims=3), dtype=torch.long)
#     return packed_sequences, batch_labels
