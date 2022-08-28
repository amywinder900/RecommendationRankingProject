import torch
from torch import nn
from torchvision import models
import copy 
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F



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


def train(epochs: int,
          model,
          optimiser,
          data_loader: dict,
          dataset_sizes: dict,
          model_save_location:str=None,
          model_load_location:str=None,
          scheduler=None,
          device:str="cpu",
          combined:bool=False):

    """
    Trains the model. The loop sends accuracy info to tensorboard.
    
    Attributes: 
    
      epochs(int): The number of epochs to run the training loop for. 
      
      model: An instantiated pytorch model which inherits from torch.nn.Module. 
      
      optimiser: An optimiser from torch.nn.optim
      
      data_loader(dict): A dictionary of dataloaders from torch.utils.data.Dataset. The keys should be "train" and "val". 
      
      dataset_sizes(dict): A dictionary of the dataset sizes corresponding to the datasets. The keys should be "train" and "val"
      
      model_save_location(str): The location of a model state to load in. Default is None. 
      
      model_load_location(str): The location of a model state to save to. Default is None. 
      
      scheduler: A learning rate scheduler from torch.optim. Default is None. 
      
      device(str): Select the processing unit to send the model to. It can either be cpu or cuda. Default is cpu.
      
      combined(bool): True if the model is processing both image features and text features. Default is False. 

    """
    writer = SummaryWriter()

    if model_load_location != None:
        model.load_state_dict(torch.load(model_load_location))

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(epochs):
        print("-"*15)
        print(f"Epoch number: {epoch}")
        print("-"*15)

        for phase in ["train", "val"]:

            if phase == "train":
                print("Training...")
                model.train()
            else:
                print("Validating...")
                model.eval()

            running_correct = 0

            for batch_index, batch in enumerate(data_loader[phase]):

                if combined:        
                    image_features, text_features, labels = batch
                    image_features, text_features, labels = image_features.to(device).float(),text_features.to(device).float(), labels.to(device)
                
                else:
                    features, labels = batch
                    features, labels = features.to(
                        device).float(), labels.to(device)

                # reset the optimiser each loop
                optimiser.zero_grad()

                # grad enabled in the training phase
                with torch.set_grad_enabled(phase == "train"):
                    if combined:
                        outputs = model(image_features, text_features)
                    else:
                        outputs = model(features)

                    _, predictions = torch.max(outputs, 1)

                    # calculate the loss
                    loss = F.cross_entropy(outputs, labels)

                    # take optimisation steps in the training fase
                    if phase == "train":
                        loss.backward()
                        optimiser.step()

                # statistics
                running_correct += torch.sum(predictions == labels.data)

            if scheduler != None and phase == 'train':
                scheduler.step()

            epoch_acc = running_correct.double() / dataset_sizes[phase]

            if phase == "train":
                writer.add_scalar("Training Accuracy", epoch_acc*100, epoch)
            else:
                writer.add_scalar("Validation Accuracy", epoch_acc*100, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

            writer.flush()

    # Load the best model weights
    model.load_state_dict(best_model_weights)

    if model_save_location != None:
        torch.save(model.state_dict(), model_save_location)

    return model

# %%
