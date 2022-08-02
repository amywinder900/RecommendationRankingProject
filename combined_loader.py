#%%
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
import requests
import random
import torch
import os
import random
from transformers import BertTokenizer
from transformers import BertModel
import pandas as pd
from torch.utils.data import Dataset

# %%


class ImageTextProductDataset(Dataset):
    def __init__(self, images_location, img_side_length, load_image_category_table, image_transform=None, max_text_length=50):

        super().__init__()

        self.img_side_length = img_side_length

        # Check image file exists
        if not os.path.exists(images_location):
            raise RuntimeError("Image Dataset not found at: ", images_location)
        self.images_location = images_location

        # Load or create the image category table
        if load_image_category_table:
            self.image_category_table = pd.read_json(
                "data/product_images.json")
        else:
            self.image_category_table = self.create_image_category_table()

        # Extract data as columns
        self.image_ids = self.image_category_table["image_id"]
        self.category_labels = self.image_category_table["main_category"]
        assert len(self.image_ids) == len(self.category_labels)

        # Create label encoder/decoder
        self.labels = self.image_category_table['main_category'].to_list()
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        self.image_transform = image_transform
        if image_transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.descriptions = self.image_category_table['product_description'].to_list()
        self.max_text_length = max_text_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

    def __getitem__(self, index):
        label = self.category_labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(self.images_location +
                           self.image_ids[index] + ".jpg")
        image = self.transform(image)

        sentence = self.descriptions[index]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length = self.max_text_length, padding = "max_length", truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.text_model(**encoded).last_hidden_state.swapaxes(1,2)
        description = description.squeeze(0)
    
        
        return image, description, label

    def __len__(self):
        return len(self.image_ids)
        

    @staticmethod
    def create_image_category_table():
        products = pd.read_json('data/products_table_clean.json')
        images = pd.read_json('data/images_table.json')
        products_images = products.merge(images, left_on='id', right_on='product_id').rename(
            columns={'id_y': 'image_id'}).drop('id_x', axis=1)
        products_images.to_json('data/product_images.json')

        return products_images
#%%
if __name__ == '__main__':
    dataset = ImageTextProductDataset("data/cleaned_images_128/", 128, True)
    print(dataset[2500])
    print(dataset.decoder[int(dataset[2500][2])])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=12,
                                             shuffle=True, num_workers=1)
    for i, (image, description, labels) in enumerate(dataloader):
        print(image)
        print(labels)
        print(description.size())
        print(image.size())
        if i == 0:
            break
# %%

