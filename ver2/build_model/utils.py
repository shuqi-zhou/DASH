import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    np.random.seed(seed)  
    random.seed(seed) 

def pad_crf_output(prediction):
    max_length = max(len(seq) for seq in prediction)
    padded_predictions = torch.full((len(prediction), max_length), -1, dtype=torch.long)
    for i, seq in enumerate(prediction):
        padded_predictions[i, :len(seq)] = torch.tensor(seq)
    return padded_predictions

def create_new_seg_and_label(y_seg, y_seg_label, final_end):
    # y_seg: [(s1, e1), (s2, e2), ...] of length n
    # y_seg_label: [1, 2, ...] of length n
    # new_seg = [0, s1, e1, s2, e2, ...] if ei < si else merge them
    # new_label = [0, 1, 0, 2], add 0 for each space between segments in y_seg

    # Initialize new_seg and new_label with the starting zero
    if len(y_seg) == 0:
        return [], [0]
    
    new_seg = [0] # new_seg start from 0
    new_label = []
    for i, (start, end) in enumerate(y_seg):
        start = int(start) 
        end = int(end)    
        label = int(y_seg_label[i])  

        if new_seg[-1] < start:
            # If the end of the previous segment is before the start of the current segment
            new_seg.extend([start, end])
            new_label.extend([0, label])
        elif new_seg[-1] == start:
            new_seg.append(end)
            new_label.append(label) 
        else:
            print("Error: Overlapping segments")
            break  

    if new_seg[-1] < final_end:
        new_seg.append(final_end)
        new_label.append(0)
    return new_seg, new_label