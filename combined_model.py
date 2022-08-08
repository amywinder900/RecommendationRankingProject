#%%
from torch import nn 
import torch
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from combined_loader import ImageTextProductDataset, create_data_loaders
import copy
from torchvision import transforms
from torch.nn import functional as F
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

    
#%%
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.text_model = TextModel(50).to(device)
        self.image_model = ImageModel().to(device)
        self.main = nn.Sequential(nn.Linear(26,13))

    def forward(self, image_features, text_features):

        image_features = self.image_model(image_features)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

#%%

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Other batch hyperparameters
validation_split = 0.2
batch_size = 8

# Create data loaders
data_loader, dataset_sizes = create_data_loaders("data/cleaned_images_128/", ImageTextProductDataset, 128, data_transforms, validation_split=validation_split, batch_size=batch_size)

#%%
def train(epochs, model, optimiser, model_save_location=None, model_load_location=None, scheduler=None, device="cpu"):
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

        running_loss = 0.0
        running_correct = 0 

        for batch_index, batch in enumerate(data_loader[phase]):
          image_features, text_features, labels = batch
          image_features, text_features, labels = image_features.to(device).float(),text_features.to(device).float(), labels.to(device)

          #reset the optimiser each loop
          optimiser.zero_grad()

          #grad enabled in the training phase
          with torch.set_grad_enabled(phase == "train"):
            
            outputs = model(image_features, text_features)
            _, predictions = torch.max(outputs, 1)
            

            # calculate the loss 
            loss = F.cross_entropy(outputs, labels)
            
            #take optimisation steps in the training fase
            if phase == "train":
              loss.backward()
              optimiser.step()


          # statistics
          running_correct += torch.sum(predictions == labels.data)
          
        if scheduler != None and phase == 'train':
          scheduler.step()


        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_correct / dataset_sizes[phase]




        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.1f}%')

        if phase == "train":
          writer.add_scalar("Training Accuracy", epoch_acc*100, epoch)
          writer.add_scalar('Training Loss', epoch_loss, epoch)
        else:
          writer.add_scalar("Validation Accuracy", epoch_acc*100, epoch)
          writer.add_scalar('Validation Loss', epoch_loss, epoch)

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

#%%
# Checks if GPU is avaliable to run on.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CombinedModel()
optimiser = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9) 
model = train(40, model, optimiser, model_save_location="/content/drive/My Drive/coding/vision_model_state.pt", device=device)
# %%
