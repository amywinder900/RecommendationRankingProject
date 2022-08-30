# %%
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import os
from transformers import BertTokenizer
from transformers import BertModel
import pandas as pd
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import numpy as np
import pickle

# %%


class ImageTextProductDataset(Dataset):
    """
    The ImageTextProductDataset object inherits its methods from the torch.utils.data.Dataset module.

    Parameters:

        images_location(str): The path to the file containing the cleaned images. 

        image_side_length(int): The side length of the images. 

        load_image_category_table(bool): If True, loads the combined image and description table. Otherwise downloads the tables from the database. Default is False.

        image_transforms(dict): A dictionary containing the the transformation or list of transformations to be done to the image. It should contain transformations for the train and validation phases using the keys "train" and "val". Default is None. 

        max_text_length(int): The number of words to consider in the description. Default is the first 50.

        decoder(dict): The dictionary which assigns each category to a number, with the key being the number and the item being the category name. Default is None.

    """

    def __init__(self,
                 images_location: str,
                 image_side_length: int,
                 load_image_category_table: bool = False,
                 image_transform=None,
                 max_text_length=50,
                 decoder=None):

        super().__init__()

        self.imgage_side_length = image_side_length

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

        if decoder == None:
            # Create label encoder/decoder
            self.labels = self.image_category_table['main_category'].to_list()
            self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
            self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

            with open('models/image_decoder.pickle', 'wb+') as f:
                decoder = pickle.dump(decoder, f)
        else:
            self.decoder = decoder
            self.encoder = {y: x for x, y in decoder.items()}

        self.image_transform = image_transform
        if image_transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.descriptions = self.image_category_table['product_description'].to_list(
        )
        self.max_text_length = max_text_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)

    def __getitem__(self, index):
        label = self.category_labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(self.images_location +
                           self.image_ids[index] + ".jpg")
        image = self.image_transform(image)

        sentence = self.descriptions[index]
        encoded = self.tokenizer.batch_encode_plus(
            [sentence], max_length=self.max_text_length, padding="max_length", truncation=True)
        encoded = {key: torch.LongTensor(value)
                   for key, value in encoded.items()}
        with torch.no_grad():
            description = self.text_model(
                **encoded).last_hidden_state.swapaxes(1, 2)
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


def create_data_loaders(images_location: str,
                        image_dataset: Dataset,
                        image_size: int,
                        image_transforms: dict,
                        load_image_category_table=True,
                        validation_split: float = 0.2,
                        batch_size: int = 16,
                        shuffle: bool = True):
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


    """
    dataset_length = len(image_dataset(
        images_location, image_size, load_image_category_table))

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
    dataset = {phase: image_dataset(images_location, image_size, load_image_category_table, image_transform=image_transforms[phase])
               for phase in ["train", "val"]}

    # Load data for each phase
    data_loader = {phase: DataLoader(dataset[phase], batch_size=batch_size, sampler=sampler[phase])
                   for phase in ["train", "val"]}

    return data_loader, dataset_sizes


# %%
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
