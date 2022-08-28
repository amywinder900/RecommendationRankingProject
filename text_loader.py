import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pickle
import pandas as pd


class ProductTextCategoryDataset(torch.utils.data.Dataset):
    """
    The ProductTextCategoryDataset object inherits its methods from the torch.utils.data.Dataset module.

    Parameters:

        categories(): The pandas series corresponding to the product category labels.

        descriptions(): The pandas series corresponding to the product descriptions. This should be in the same order as the category labels.

        max_length(int): The number of words to consider in the description. Default is the first 50.

        decoder(dict): The dictionary which assigns each category to a number, with the key being the number and the item being the category name. Default is None.


    """

    def __init__(self,
                 categories: pd.Series,
                 descriptions: pd.Series,
                 max_length: int = 50,
                 decoder: dict = None):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.max_length = max_length

        self.labels = categories.to_list()
        # self.descriptions = descriptions.to_list()

        #TODO fix this chokepoint 
        self.descriptions = [self.embed_text(
            description) for description in descriptions.to_list()]

        self.num_classes = len(set(self.labels))

        if decoder == None:
            # Create label encoder/decoder
            self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
            self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

            with open('models/text_decoder.pickle', 'wb+') as f:
                decoder = pickle.dump(decoder, f)
        else:
            self.decoder = decoder
            self.encoder = {y: x for x, y in decoder.items()}

    def __getitem__(self, index):

        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        description = self.descriptions[index]
        description = description.squeeze(0)

        return description, label

    def embed_text(self, description):
        encoded = self.tokenizer.batch_encode_plus(
            [description], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key: torch.LongTensor(value)
                   for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(
                **encoded).last_hidden_state.swapaxes(1, 2)
        return description


def create_data_loaders(text_dataset: Dataset,
                        categories: pd.Series,
                        descriptions: pd.Series,
                        max_length: int = None,
                        validation_split: float = 0.2,
                        batch_size: int = 16,
                        shuffle: bool = True,
                        decoder: dict = None):
    """
    This function creates the dataloaders with a training and validation split. 

    Parameters:

        text_dataset(Dataset): The dataset object which prepares the text and the category label. 

        categories(pd.Series): The pandas series corresponding to the product category labels.

        descriptions(pd.Series): The pandas series corresponding to the product descriptions. This should be in the same order as the category labels.

        max_length(int): The number of words to consider in the description. Default is the first 50.

        validation_split(float): The proportion of the dataset which should be used for validation. Default is 0.2

        batch_size(int): The batch size to process the images in. 

        shuffle(bool): Whether or not to shuffle the order of the images. Default is true.

      decoder(dict): The dictionary which assigns each category to a number, with the key being the number and the item being the category name. Default is None.

    Returns:
        data_loader(dict): The dictionary of dataloaders with keys "train" and "val". 

        dataset_sizes(dict): The dictionary of dataset sizes with keys "train" and "val". 


    """
    dataset_length = len(text_dataset(categories, descriptions))

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
    dataset = {phase: text_dataset(categories, descriptions, max_length, decoder=decoder)
               for phase in ["train", "val"]}

    # Load data for each phase
    data_loaders = {phase: DataLoader(dataset[phase], batch_size=batch_size, sampler=sampler[phase])
                    for phase in ["train", "val"]}

    return data_loaders, dataset_sizes
