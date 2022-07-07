from download_from_database import DownloadTables
import pandas as pd
import torch
import numpy as np
from PIL import Image
from alive_progress import alive_bar
from torchvision.transforms import ToTensor
from clean_tabular import CleanTabular
from os.path import isfile


class PrepareData:
    """
    Class to process images for a classifcation model.

        Attributes:
            crediential_location(str): Optionally, the location of the yaml file containing the database credentials. It should contain DATABASE_TYPE, DBAPI, ENDPOINT, USER, PASSWORD, PORT, DATABASE
    """

    def __init__(self, credentials_location: str = None):
        """
        See help(PrepareData)
        """
        product_df, image_df = self.retrieve_dataframes(credentials_location)

        self.product_df = product_df
        self.image_df = image_df

    def retrieve_dataframes(self, credentials_location: str = None):
        """
        Retrives the Dataframes..


        Attributes:
            crediential_location(str): Optionally, the location of the yaml file containing the database credentials. It should contain DATABASE_TYPE, DBAPI, ENDPOINT, USER, PASSWORD, PORT, DATABASE

        Returns:
            product_df(DataFrame): The clean DataFrame from the products tables. 
            image_df (DataFrame): The clean DataFrame from the images table. 


        """
        # if credentials location is given, attempt to download the tables
        if credentials_location != None:
            downloader = DownloadTables(credentials_location)
            downloader.download_table("images")
            downloader.download_table("products")

        # check the appropriate products table exists

        # product_table_location = "data/products_table_clean.json"
        product_table_location = "data/products_table_logistic_regression.json"
        if isfile(product_table_location) == False:
            print("Creating ", product_table_location)
            data_location = "data/products_table.json"
            product_data = pd.read_json(data_location)

            cleaner = CleanTabular(product_data)
            cleaner.create_main_category_column()
            product_data = cleaner.get_product_df()
            product_data.to_json(product_table_location)

        # read the data into dataframes
        product_df = pd.read_json(product_table_location)
        image_df = pd.read_json("data/images_table.json")

        return product_df, image_df

    def retrieve_image_category(self, image_name: str) -> str:
        """
        Given the ID of an image, retrieves the category. 

        Attributes:
            image_name(str): the name of when 
        """
        image_row = self.image_df.loc[self.image_df["id"] == image_name]
        product_id = image_row.iloc[0]["product_id"]

        product_row = self.product_df.loc[self.product_df["id"] == product_id]
        category = product_row.iloc[0]["main_category"]

        return category

    @staticmethod
    def image_to_array(image_location):
        """
        Converts an image to a numpy array, 
        """
        image = Image.open(image_location)
        image = ToTensor()(image)
        image = torch.flatten(image)
        return image.numpy()

    def convert_to_image_array(self, image, images_folder):
        image_category = self.retrieve_image_category(image)
        image_location = images_folder + "/" + image + ".jpg"
        image_array = self.image_to_array(image_location)
        return image_array, image_category

    # def create_dict_of_categories(self):
    #     categories = set(self.product_df["main_category"])
    #     categories_dict = {k: (v - 1) for v, k in enumerate(categories)}
    #     self.categories_dict = categories_dict
    #     return categories_dict

    def form_arrays(self, images_folder, image_size: int, n: int = None):

        images = list(self.image_df["id"])

        # if the number of images hasn't been specified, look at all images
        if n == None:
            n = len(images)

        # sets up the arrays
        # array_size is based on a square image with three channels
        array_size = (image_size**2)*3
        X = np.zeros((n, array_size))
        y = np.zeros(n)

        # #set up a dictionary assigning categories to integers
        # self.create_dict_of_categories()

        self.encoder = {y: x for (x, y) in enumerate(set(self.product_df["main_category"]))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.product_df["main_category"]))}

        for index in range(n):
            image = images[index]
            try:
                features, label = self.convert_to_image_array(
                    image, images_folder)
                X[index, :] = features
                y[index] = self.encoder[label]

            except:
                # TODO deal with this because it might be affecting the model
                X[index, :] = np.zeros(array_size)
                y[index] = 0

        self.X = X
        self.y = y
        return X, y

    def save_to_files(self, X_file_location, y_file_location):
        np.save(X_file_location, self.X)
        np.save(y_file_location, self.y)
        np.save("data/encoder.npy", self.encoder)
        np.save("data/decoder.npy", self.decoder)
        return None


if __name__ == "__main__":

    image_size = 124
    credentials_location = ".gitignore/credentials_for_marketplace.yml"
    images_folder = "data/cleaned_images_" + str(image_size)
    X_array_folder = "data/X_for_image_"+ str(image_size) + ".npy"
    y_array_folder = "data/y_for_image_"+ str(image_size)+".npy"
    
    pipeline = PrepareData()
    product_df, image_df = pipeline.retrieve_dataframes(credentials_location)
    X, y = pipeline.form_arrays(images_folder, image_size, n=10 )
    pipeline.save_to_files(X_array_folder, y_array_folder)
    pass
