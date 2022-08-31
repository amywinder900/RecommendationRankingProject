# %%
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from transformers import BertModel
# %%

class ImageProcessor:
    """
    This class can be called on a PIL Image. It performs the necessary transformations so a model can be called on an image. 
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  
        ])

    def __call__(self, image:Image):
        image = self.transform(image)
        # Add a dimension to the image
        image = image[None, :, :, :]
        return image

class TextProcessor:
    """
    This class can be called on a string. It performs the necessary transformations so a model can be called on an text desciption. 
    """

    def __init__(self,
                 max_length: int = 50):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.max_length = max_length

    def __call__(self, text:str):
        encoded = self.tokenizer.batch_encode_plus(
            [text], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key: torch.LongTensor(value)
                   for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(
                **encoded).last_hidden_state.swapaxes(1, 2)
        return description

# %%
if __name__ == "__main__":
    import PIL
    import torch
    from neural_network_models import ImageModel
    
    # uses cpu to load model
    device = torch.device('cpu')

    model = ImageModel()
    model.load_state_dict(torch.load(
        "models/vision_model_state.pt", map_location=device))
    model.eval()

    image_path = "data/cleaned_images_128/0a2d446d-f9ac-4715-992d-1bb30017e44d.jpg"

    image = ImageProcessor()(PIL.Image.open(image_path))

    model.predict(image)
# %%

if __name__=="__main__":
    from neural_network_models import TextModel
    import pickle
    import torch

    device = torch.device('cpu')
    with open('models/decoder.pickle', 'rb') as f:
        decoder = pickle.load(f)


    model = TextModel(50, decoder)
    model.load_state_dict(torch.load(
        "models/nlp_model_state.pt", map_location=device))
    model.eval()

    text = TextProcessor()(input("Enter description: "))
    model.predict_classes(text)