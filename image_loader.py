import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image


class ProductImageCategoryDataset(Dataset):
    """
    The ProductImageCategoryDataset object inherits its methods from the torch.utils.data.Dataset module.
    It loads images from a numpy array. The images should be processed so that the height and width is the same. 

    Parameters:

        images_location(str): The folder location containing the cleaned images.

        img_side_length(int): The side length of the images. 

        load_image_category_table(bool): If true, the class loads the image category table from data/image_category_table.json or if False it will create a new one. Default value is False.

        transform: The transformation or list of transformations to be done to the image. If no transform is passed, the class will do a generic transformation to resize, convert it to a tensor, and normalize the numbers.

        decoder(dict): The dictionary which assigns each category to a number, with the key being the number and the item being the category name. Default is None.


    """

    def __init__(self,
                 images_location: str,
                 img_side_length: int,
                 load_image_category_table: bool = False,
                 transform: transforms = None,
                 decoder:dict=None):

        super().__init__()

        self.img_side_length = img_side_length

        # Check image file exists
        if not os.path.exists(images_location):
            raise RuntimeError("Image Dataset not found at: ", images_location)
        self.images_location = images_location

        # Load or create the image category table
        if load_image_category_table:
            self.image_category_table = pd.read_json(
                "data/image_category_table.json")
        else:
            self.image_category_table = self.create_image_category_table()

        # Extract data as columns
        self.image_ids = self.image_category_table["image_id"]
        self.category_labels = self.image_category_table["main_category"]
        assert len(self.image_ids) == len(self.category_labels)

        if decoder == None:
            # Create label encoder/decoder
            self.labels = self.image_category_table['main_category'].to_list()
            self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
            self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
        else: 
            self.decoder = decoder
            self.encoder = {y:x for x, y in decoder.items()}

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

        label = self.category_labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(self.images_location +
                           self.image_ids[index] + ".jpg")
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_ids)


def create_data_loaders(images_location: str,
                        image_dataset:Dataset,  
                        image_size:int, 
                        image_transforms:dict, 
                        validation_split:float=0.2, 
                        batch_size:int=16, 
                        shuffle:bool=True,
                        decoder:dict=None):
    """
    This function creates the dataloaders with a training and validation split. 

    Parameters:

      images_location(str): The path to the file containing the cleaned images. 

      image_dataset(Dataset): The dataset object which prepares the image and the category label. 

      image_side_length(int): The side length of the images. 

      image_transforms(dict): A dictionary containing the the transformation or list of transformations to be done to the image. It should contain transformations for the train and validation phases using the keys "train" and "val". 

      validation_split(float): The proportion of the dataset which should be used for validation. Default is 0.2

      batch_size(int): The batch size to process the images in. 

      shuffle(bool): Whether or not to shuffle the order of the images. Default is true.

    Returns:
        data_loader(dict): The dictionary of dataloaders with keys "train" and "val". 

        dataset_sizes(dict): The dictionary of dataset sizes with keys "train" and "val". 


    """
    dataset_length = len(image_dataset(images_location, image_size))

    # find split
    split_indice = int(np.floor(validation_split * dataset_length))

    # create list of indices and shuffle
    indices = list(range(dataset_length))
    if shuffle:
        np.random.shuffle(indices)

    # Create dictionary of dataset sizes
    dataset_sizes = {"train": dataset_length - int(np.floor(validation_split * dataset_length)),
                     "val": int(np.floor(validation_split * dataset_length))}

    # Create sampler
    sampler = {"train": SubsetRandomSampler(indices[split_indice:]),
               "val": SubsetRandomSampler(indices[:split_indice])}

    # Form datasets
    dataset = {phase: image_dataset(images_location, image_size, transform=image_transforms[phase], decoder=decoder)
               for phase in ["train", "val"]}

    # Load data for each phase
    data_loaders = {phase: DataLoader(dataset[phase], batch_size=batch_size, sampler=sampler[phase])
                   for phase in ["train", "val"]}

    return data_loaders, dataset_sizes

