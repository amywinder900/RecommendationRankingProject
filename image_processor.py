#%%
import torch
from torchvision import models
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
#%%
class resnet50CNN(torch.nn.Module): 
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
    
  def predict(self, inp):
      with torch.no_grad():
        x = self.forward(inp)
        return x
#%%

def image_to_array(image_location, image_transform=None, img_side_length=128):
    """
    Converts an image to a numpy array, 
    """
    #TODO refactor the image processing
    image = Image.open(image_location)
    image = ToTensor()(image)
    image = torch.flatten(image)
    image = torch.tensor(image)
    image = image.reshape(3, img_side_length, img_side_length)
    image = transforms.ToPILImage()(image)
    if image_transform:
        image = image_transform(image)
    # image_array = torch.unsqueeze(image, 0)
    return image
    
#%%
# uses cpu to load model
device = torch.device('cpu')

model = resnet50CNN()
model.load_state_dict(torch.load("models/vision_model_state.pt", map_location=device))
model.eval()
#%%
image_path = "data/cleaned_images_128/0a2d446d-f9ac-4715-992d-1bb30017e44d.jpg"

image_transformation =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#%%
image = image_to_array(image_path)


# %%
model.predict(image)
# %%
