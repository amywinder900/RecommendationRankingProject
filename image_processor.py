# %%
import torch
from torchvision import models
from torchvision import transforms
# %%


class resnet50CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.resnet50(pretrained=True)

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

    def predict(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return x
# %%


class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # is this right?
        ])

    def __call__(self, image):

        image = self.transform(image)
        # Add a dimension to the image
        image = image[None, :, :, :]
        return image


# %%
if __name__ == "__main__":
    import PIL
    # uses cpu to load model
    device = torch.device('cpu')

    model = resnet50CNN()
    model.load_state_dict(torch.load(
        "models/vision_model_state.pt", map_location=device))
    model.eval()

    image_path = "data/cleaned_images_128/0a2d446d-f9ac-4715-992d-1bb30017e44d.jpg"

    image = ImageProcessor()(PIL.Image.open(image_path))

    model.predict(image)
# %%
