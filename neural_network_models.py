import torch
from torch import nn
from torchvision import models


class TextModel(torch.nn.Module):
    def __init__(self,  embedding_size, decoder=None):
        super().__init__()

        self.layers = torch.nn.Sequential(
            nn.Conv1d(embedding_size, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(98304, 256),
            nn.ReLU(),
            nn.Linear(256, 13)
        )

        self.decoder = decoder

    def forward(self, X):

        # X = self.embedding(X)
        X = X.transpose(2, 1)
        X = self.layers(X)
        return X


def predict(self, image):
    with torch.no_grad():
        x = self.forward(image)
        return x


def predict_prob(self, image):
    with torch.no_grad():
        x = self.forward(image)
        return torch.softmax(x, dim=1)


def predict_class(self, image):
    if self.decoder == None:
        raise Exception("Decoder was not passed when instantiating model.")
    with torch.no_grad():
        x = self.forward(image)
        return self.decoder(int(torch.argmax(x, dim=1)))


class ImageModel(torch.nn.Module):
    def __init__(self, decoder=None):
        super().__init__()
        self.features = models.resnet50(pretrained=True)
        self.decoder = decoder

        # Freezes the first layers of resnet so that only the final few layers are being trained
        for i, parameter in enumerate(self.features.parameters()):
            if i < 47:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

        # Replace the final layer with my own layers.
        self.features.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 13),
        )

    def forward(self, x):
        x = self.features(x)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_prob(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)

    def predict_class(self, image):

        if self.decoder == None:
            raise Exception("Decoder was not passed when instantiating model.")
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder(int(torch.argmax(x, dim=1)))

# %%


class CombinedModel(nn.Module):
    def __init__(self, device="cpu", decoder=None):
        super(CombinedModel, self).__init__()

        self.text_model = TextModel(50).to(device)
        self.image_model = ImageModel().to(device)
        self.main = nn.Sequential(nn.Linear(26, 13))

        self.decoder = decoder

    def forward(self, image_features, text_features):

        image_features = self.image_model(image_features)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_prob(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)

    def predict_class(self, image):
        if self.decoder == None:
            raise Exception("Decoder was not passed when instantiating model.")

        with torch.no_grad():
            x = self.forward(image)
            return self.decoder(int(torch.argmax(x, dim=1)))
