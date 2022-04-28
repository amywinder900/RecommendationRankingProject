import pandas as pd 
from pandas import DataFrame
import numpy as np

class CleanTabular:
    """ This class is used to clean tabular data.
    Attributes:
        product_df (DataFrame): the updated dataframe.
    """
    def __init__(self, product_df:DataFrame):
        """
        See help(CleanTabular)
        """
        self.product_df = product_df


    def remove_null(self) -> None:
        """
        Removes the rows with null values. 
        
        """
        temp_df = self.product_df
        #replaces the "N/A" characters with numpy null values
        temp_df = temp_df.replace("N/A", np.nan)

        temp_df.dropna(inplace=True)
        
        self.product_df = temp_df
        return None

    def clean_prices(self) -> None:
        """
        Cleans the prices and converts to integer. 
        """
        self.product_df["price"] = self.product_df["price"].str.strip("£")
        self.product_df["price"] = self.product_df["price"].str.replace(",","")
        self.product_df["price"] = self.product_df["price"].astype("float64")
        
        return None

    def create_main_category_column(self)-> None:
        """
        Creates a column which displays the main category. 

        """
        self.product_df["main_category"] = self.product_df["category"].apply(lambda x: x.split("/")[0].strip())
        
        return None

    def add_county_column(self) -> None :
        """
        Adds a column which shows the county. 
        
        """
        self.product_df["county"] = self.product_df["location"].apply(lambda x: x.split(",")[-1].strip())
        return None
    
    def remove_duplicates(self) -> None:
        """
        Removes the duplicates from the dataframe. 
        """
        
        columns = [ "product_name", "product_description", "location"]
        self.product_df.drop_duplicates(subset=columns, keep="first", )
        return None

    def get_product_df(self) -> DataFrame:
        """
        Retrieves the dataframe. 

        Returns:
            self.product_df(DataFrame): The dataframe for this instance of the class.
        """

        return self.product_df

if __name__ == "__main__":
    #read data from file
    print("Reading data from data/products_table.json")
    product_data = pd.read_json("data/products_table.json")
    #perform cleaning
    print("Cleaning data")
    cleaner = CleanTabular(product_data)
    cleaner.remove_null()
    cleaner.clean_prices()
    cleaner.add_county_column()
    cleaner.create_main_category_column()
    cleaner.remove_duplicates()
    #retrieve the data from class
    product_data = cleaner.get_product_df()
    #send the data to file
    print("Sending cleaned data to data/products_table_clean.json")
    product_data.to_json("data/products_table_clean.json")


