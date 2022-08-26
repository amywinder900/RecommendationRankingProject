# %%
from typing import Text
import torch
from torch import nn
from transformers import BertTokenizer
from transformers import BertModel
import pickle
import numpy as np
# %%


class CNN(torch.nn.Module):
    def __init__(self,  embedding_size, decoder=None):
        super().__init__()

        self.decoder = decoder

        self.layers = torch.nn.Sequential(
            nn.Conv1d(embedding_size, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(98304, 256),
            nn.ReLU(),
            nn.Linear(256, 13)
        )

    def forward(self, X):

        # X = self.embedding(X)
        X = X.transpose(2, 1)
        X = self.layers(X)
        return X

    def predict(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return x

    def predict_proba(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return torch.softmax(x, dim=1)

    def predict_classes(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return self.decoder[int(torch.argmax(x, dim=1))]

# %%


class TextProcessor:
    def __init__(self,
                 max_length: int = 50):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.max_length = max_length

    def __call__(self, text):
        encoded = self.tokenizer.batch_encode_plus(
            [text], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key: torch.LongTensor(value)
                   for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(
                **encoded).last_hidden_state.swapaxes(1, 2)
        return description
# %%
if __name__=="__main__":
    device = torch.device('cpu')
    with open('models/decoder.pickle', 'rb') as f:
        decoder = pickle.load(f)


    model = CNN(50, decoder)
    model.load_state_dict(torch.load(
        "models/nlp_model_state.pt", map_location=device))
    model.eval()

    text = TextProcessor()(input("Enter description: "))
    model.predict_classes(text)
# %%
