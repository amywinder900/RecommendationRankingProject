import pandas as pd 
from pandas import DataFrame

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

    @staticmethod
    def filter_rows_by_values(product_df:DataFrame, columns:list, values:list) -> DataFrame:
        """
        For a given dataframe, returns a dataframe which contains given values in selected columns.
        
        Arguments:
            product_df(DataFrame): The dataframe to work with. 
            columns(array): The columns to check.
            values(array): The values to check for.
        
        Returns:
            product_df(DataFrame):The resulting DataFrame. 

        """
        product_df = product_df[~product_df[columns].isin(values)]
        return product_df

    def remove_null(self) -> None:
        """
        Removes the rows with null values. 
        
        """
        self.product_df.dropna(inplace=True)
        #remove rows with "N/A" in any column
        self.product_df = CleanTabular.filter_rows_by_values(self.product_df, self.product_df.columns, ["N/A"])
        return None

    def clean_prices(self) -> None:
        """
        Cleans the prices and converts to integer. 
        """
        self.product_df["price"] = self.product_df["price"].str.strip("£")
        self.product_df["price"] = self.product_df["price"].str.replace(",","")
        self.product_df["price"] = self.product_df["price"].astype("float64")
        
        return None

    @staticmethod
    def extract_county(location:str) -> str:
        """
        Retrieves the county/greater area from the location.
        
        Arguments:
            location(str): A location with the format "city, county"
        Returns:
            county(str): The county from the location.
        
        """
        county = location.split(",")[-1].strip()
        return county

    def add_county_column(self) -> None :
        """
        Adds a column which shows the county. 
        
        """
        self.product_df["county"] = self.product_df["location"].apply(CleanTabular.extract_county)
        return None
    
    def remove_duplicates(self) -> None:
        """
        Removes the duplicates from the dataframe. 
        """
        
        columns = [ "product_name", "product_description", "location"]
        self.product_df.drop_duplicates(subset=columns, keep="first", )
        return None

    def get_product_df(self):
        return self.product_df

if __name__ == "__main__":
    #read data from file
    product_data = pd.read_json("products_table.json")
    #perform cleaning
    cleaner = CleanTabular(product_data)
    cleaner.remove_null()
    cleaner.clean_prices()
    #TODO fix bug with county column
    # cleaner.add_county_column()
    cleaner.remove_duplicates()
    #retrieve the data from class
    product_data = cleaner.get_product_df()
    #send the data to file
    product_data.to_json("products_table_clean.json")

