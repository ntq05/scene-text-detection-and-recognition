import torch
from torch.nn import functional as F

def encode(label, char_to_idx, max_label_len):
    encoded_labels = torch.tensor(
        [char_to_idx[char] for char in label], dtype=torch.long
    ) 

    label_len = len(encoded_labels)
    lengths = torch.tensor(label_len, dtype = torch.long)
    padded_labels = F.pad(encoded_labels, (0, max_label_len - label_len), value=0)

    return padded_labels, lengths

def decode_label(encoded_sequences, idx_to_char, pad_idx=0):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded = []

        for token in seq:
            token = token.item()

            if token == 0:
                continue

            if token == pad_idx:
                continue

            decoded.append(idx_to_char[token])

        decoded_sequences.append("".join(decoded))

    return decoded_sequences

def decode_prediction(encoded_sequences, idx_to_char, blank_idx=1):
    decoded_sequences = []

    for seq in encoded_sequences:

        decoded = []
        prev_token = blank_idx

        for token in seq:

            token = token.item()

            if token == 0:
                continue

            if token == blank_idx:
                prev_token = token
                continue

            if token != prev_token:
                decoded.append(idx_to_char[token])

            prev_token = token

        decoded_sequences.append("".join(decoded))

    return decoded_sequences