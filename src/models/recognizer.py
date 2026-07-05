import torch
from utils.tokenizer import decode_prediction
from .architectures.crnn import CRNN
from ..config.settings import HIDDEN_SIZE, N_LAYERS, DROPOUT_PROB, UNFREEZE_LAYERS

class Recognizer:
    def __init__(self, model_path, data_transforms, device):
        self.chars = "0123456789abcdefghijklmnopqrstuvwxyz-"
        self.vocab_size = len(self.chars)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(self.chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.data_transforms = data_transforms
        self.device = device
        self.model = CRNN(
            vocab_size = self.vocab_size,
            hidden_size = HIDDEN_SIZE,
            n_layers = N_LAYERS,
            dropout = DROPOUT_PROB,
            unfreeze_layers = UNFREEZE_LAYERS
        ).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()


    def recognize(self, image):

        transformed_image = self.data_transforms(image)
        transformed_image = transformed_image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(transformed_image).detach().cpu()

        text = decode_prediction(logits.permute(1, 0, 2).argmax(2), self.idx_to_char)
        return text[0]