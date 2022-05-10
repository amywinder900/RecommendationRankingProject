import pandas as pd
from pandas import DataFrame
import numpy as np
import requests
from tqdm import tqdm


class CleanTabular:
    """ This class is used to clean tabular data.

    Attributes:
        product_df (DataFrame): the updated dataframe.
    """

    def __init__(self, product_df: DataFrame):
        """
        See help(CleanTabular)
        """
        self.product_df = product_df

    # def drop_columns(self, columns:list) -> None:
    #     """
    #     Takes in a list of columns and drops them from the dataframe.

    #     Attributes:
    #         columns(list): The list of column names.

    #     """
    #     self.product_df = self.product_df.drop(columns, axis=1)

    #     return None

    def remove_null(self) -> None:
        """
        Removes the rows with null values. 

        """
        print("Removing null data.")
        temp_df = self.product_df
        # replaces the "N/A" characters with numpy null values
        temp_df = temp_df.replace("N/A", np.nan)

        temp_df.dropna(inplace=True)

        self.product_df = temp_df
        return None

    def clean_prices(self, max_price: int = 10000, min_price: int = 0.1) -> None:
        """
        Cleans the prices and converts to integer. 

        Attributes:
            max_price(int): The maximum allowable price for the data. Default is £10,000. 
            min_price(int): The minimum allowable price for the data. Default is £0.10. 
        """
        print("Cleaning prices.")
        self.product_df["price"] = self.product_df["price"].str.strip("£")
        self.product_df["price"] = self.product_df["price"].str.replace(
            ",", "")
        self.product_df["price"] = self.product_df["price"].astype("float64")
        
        # Remove outliers

        self.product_df = self.product_df[self.product_df["price"] > min_price]
        self.product_df = self.product_df[self.product_df["price"] < max_price]
        return None

    @staticmethod
    def retrieve_longitude_and_latitude(location: str):
        """
        Given a location of the form "city, district" or similar, retrieves the longitude and latitude.

        Returns:
            latitude(float): The latutude of a location. If the location can't be found then the function returns np.nan
            longitude(float): The longitude of a location. If the location can't be found then the function returns np.nan
        """
        global pbar

        city = location.split(",")[0].strip()
        district = location.split(",")[0].strip()

        url = "https://nominatim.openstreetmap.org/?addressdetails=1&q=" + \
            city + "+" + district + "&format=json&limit=1"

        response = requests.get(url).json()

        pbar.update(1)

        try:
            latitude = float(response[0]["lat"])
            longitude = float(response[0]["lon"])
        except IndexError:
            print("Can't find ", location, " skipping")
            return np.nan, np.nan
        return latitude, longitude

    def create_longitude_and_latitude_columns(self) -> None:
        """
        Adds columns for longitude and latitude in the DataFrame. 
        """
        global pbar

        print("Creating longitude and latitude columns. This step may take a long time.")
        # retrieve the longitudes and latitudes for each row
        with tqdm(total=self.product_df.shape[0]) as pbar:
            longitude, latitude = np.vectorize(
                CleanTabular.retrieve_longitude_and_latitude)(self.product_df["location"])

        # send the longitudes and latitudes to the DataFrame
        self.product_df["longitude"] = longitude.tolist()
        self.product_df["latitude"] = latitude.tolist()

        # remove any rows with null values
        self.product_df.dropna(inplace=True)

        return None

    def create_main_category_column(self) -> None:
        """
        Creates a column which displays the main category. 

        """
        self.product_df["main_category"] = self.product_df["category"].apply(
            lambda x: x.split("/")[0].strip())

        return None

    def encode_categorical_data(self, column_name: str) -> None:
        """
        Given the name of a column, this method encodes the categorical data by creating a new column for each category with binary True/False values. 
        The method drops the first encoded column so that the data is linearly independent. 

        Attributes
            column_name(str): The name of the column containing thecategorical data.
        """
        print("Encoding categorical data for ", column_name)
        # clean the column name
        self.product_df[column_name] = self.product_df[column_name].str.lower(
        ).replace('[^0-9a-zA-Z]+', '_', regex=True)
        # create the category encodings
        category_encodings = pd.get_dummies(
            self.product_df[column_name], prefix=column_name, drop_first=True)
        # merge the dataframes
        self.product_df = pd.concat(
            [self.product_df, category_encodings], axis=1)
        return None

    def remove_duplicates(self) -> None:
        """
        Removes the duplicates from the dataframe. 
        """
        print("Removing duplicates.")
        columns = ["product_name", "product_description", "location"]
        self.product_df.drop_duplicates(subset=columns, keep="first", )

        return None

    def prepare_data(self):
        """
        Method to run the standard data cleaning steps so that it is ready for basic machine learning algorithms. 
        """

        CleanTabular.remove_duplicates(self)
        CleanTabular.remove_null(self)
        CleanTabular.clean_prices(self)
        CleanTabular.create_longitude_and_latitude_columns(self)
        CleanTabular.create_main_category_column(self)
        CleanTabular.encode_categorical_data(self, "main_category")
        return None 

    def get_product_df(self) -> DataFrame:
        """
        Retrieves the dataframe. 

        Returns:
            self.product_df(DataFrame): The dataframe for this instance of the class.
        """

        return self.product_df


if __name__ == "__main__":
    # file locations
    data_location = "data/products_table.json"
    save_location = "data/products_table_for_linear_regression.json"

    # read data from file
    print("Reading data from ", data_location)
    product_data = pd.read_json(data_location)

    # perform cleaning
    print("Cleaning data")
    cleaner = CleanTabular(product_data)
    cleaner.prepare_data()

    # retrieve the data from class
    product_data = cleaner.get_product_df()

    # send the data to file
    print("Sending cleaned data to ", save_location)
    product_data.to_json(save_location)
