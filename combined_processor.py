
#%%
from torch import nn 
import torch
from torchvision import models
from combined_loader import ImageTextProductDataset, create_data_loaders
import copy
from torchvision import transforms
from torch.nn import functional as F
from transformers import BertTokenizer
from transformers import BertModel
import pickle
#%%
class TextModel(torch.nn.Module):
  def __init__(self,  embedding_size):
    super().__init__()

    self.layers = torch.nn.Sequential(
                                  nn.Conv1d(embedding_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2), 
                                  nn.Flatten(),
                                  nn.Linear(98304 , 256),
                                  nn.ReLU(),
                                  nn.Linear(256,13)
                                  )


  def forward(self, X):
    # X = self.embedding(X)
    X = X.transpose(2, 1)
    X = self.layers(X)
    return X


class ImageModel(torch.nn.Module): 
  def __init__(self):
    super().__init__()
    self.features = models.resnet50(pretrained=True)
    
    # Freezes the first layers of resnet so that only the final few layers are being trained
    for i, parameter in enumerate(self.features.parameters()):
            if i < 47:
                parameter.requires_grad=False
            else:
                parameter.requires_grad=True

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

    

class CombinedModel(nn.Module):
    def __init__(self, device="cpu", text_model_load_location=None, image_model_load_location=None, decoder=None):
        super(CombinedModel, self).__init__()

        self.decoder = decoder

        self.text_model = TextModel(50).to(device)
        # for param in self.text_model.parameters():
        #   param.requires_grad = False
        if text_model_load_location != None:
          self.text_model.load_state_dict(torch.load(text_model_load_location))


        self.image_model = ImageModel().to(device)
        # for param in self.image_model.parameters():
        #   param.requires_grad = False
        if image_model_load_location != None:
          self.image_model.load_state_dict(torch.load(image_model_load_location))

        self.main = nn.Sequential(nn.Linear(26,13))


    def forward(self, image_features, text_features):

        image_features = self.image_model(image_features)
        text_features = self.text_model(text_features)

        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)

        return combined_features

    def predict(self,image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return x

    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return torch.softmax(x, dim=1)

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return self.decoder[int(torch.argmax(x, dim=1))]

#%%
class CombinedProcessor:
    def __init__(self,
                 max_text_length: int = 50):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.max_text_length = max_text_length

        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # is this right?
        ])
    def __call__(self,image, text):
        image = self.image_transform(image)
        # Add a dimension to the image
        image = image[None, :, :, :]

        encoded = self.tokenizer.batch_encode_plus(
            [text], max_length=self.max_text_length, padding='max_length', truncation=True)
        encoded = {key: torch.LongTensor(value)
                   for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(
                **encoded).last_hidden_state.swapaxes(1, 2)

        return image, description
#%%

if __name__ == "__main__":
    import PIL
    # uses cpu to load model
    device = torch.device('cpu')

    with open('models/decoder.pickle', 'rb') as f:
        decoder = pickle.load(f)

    model = CombinedModel(decoder=decoder)
    model.load_state_dict(torch.load(
        "models/combined_model_state.pt", map_location=device))
    model.eval()

    text = "This is a very impressive chair."
    image_path = "data/cleaned_images_128/0a2d446d-f9ac-4715-992d-1bb30017e44d.jpg"

    image, text = CombinedProcessor()(PIL.Image.open(image_path), text)

    print(model.predict_classes(image, text))
# %%
