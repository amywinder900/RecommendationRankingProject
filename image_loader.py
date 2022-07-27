#%%
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from os.path import isfile
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms
from PIL import Image


class ProductImageCategoryDataset(Dataset):
    """
    The ProductImageCategoryDataset object inherits its methods from the torch.utils.data.Dataset module.
    It loads images from a numpy array. The images should be processed so that the height and width is the same. 

    Parameters:

    X(numpy.ndarray): The image data as a numpy array.

    y(numpy.ndarray): The image category data as a numpy array. 

    img_side_length(int): The side length of the images. 

    transform: The transformation or list of transformations to be done to the image. If no transform is passed, the class will do a generic transformation to resize, convert it to a tensor, and normalize the numbers


    """

    def __init__(self,
                 images_location: str,
                 img_side_length: int,
                 load_image_category_table: bool = False,
                 transform: transforms = None):

        super().__init__()

        self.img_side_length = img_side_length
        
        # Check image file exists
        if not os.path.exists(images_location):
            raise RuntimeError("Image Dataset not found at: ", images_location)
        self.images_location = images_location

        # Load or create the image category table
        if load_image_category_table:
            self.image_category_table = pd.read_json("data/image_category_table.json")
        else:
            self.image_category_table = self.create_image_category_table()

        # Extract data as columns
        self.image_ids = self.image_category_table["image_id"]
        self.category_labels = self.image_category_table["main_category"]
        assert len(self.image_ids) == len(self.category_labels)
        
        #Create label encoder/decoder
        self.labels = self.image_category_table['main_category'].to_list()
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @staticmethod
    def create_image_category_table():
        products = pd.read_json('data/products_table_clean.json')
        images = pd.read_json('data/images_table.json')
        products_images = products.merge(images, left_on='id', right_on='product_id').rename(
            columns={'id_y': 'image_id'}).drop('id_x', axis=1)
        products_images.to_json('data/product_images.json')
        return products_images

    def __getitem__(self, index: int):
        
        # features = self.X[index]
        # features = torch.tensor(features)
        # features = features.reshape(
        #     3, self.img_side_length, self.img_side_length)
        # features = transforms.ToPILImage()(features)
        # if self.transform:
        #     features = self.transform(features)

        label = self.category_labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(self.images_location + self.image_ids[index] +".jpg")
        return image, label

    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    dataset = ProductImageCategoryDataset("data/cleaned_images_128/", 128)
    print(dataset[0])
    print(dataset.decoder[int(dataset[0][1])])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=12,
                                             shuffle=True, num_workers=1)
    for i, (data, labels) in enumerate(dataloader):
        print(data)
        print(labels)
        print(data.size())
        if i == 0:
            break

# %%

# %%
