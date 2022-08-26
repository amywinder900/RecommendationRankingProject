from yaml import safe_load as yaml_load
from sqlalchemy import create_engine
import pandas as pd
from os.path import isfile


class DownloadTables:
    """
    Given a table name, this class is for downloading tables from the database.

    Attributes:
        crediential_location(str): The location of the yaml file containing the database credentials. It should contain DATABASE_TYPE, DBAPI, ENDPOINT, USER, PASSWORD, PORT, DATABASE

    """

    def __init__(self, credentials_location: str) -> None:
        """
        See help(DownloadTables)
        """

        with open(credentials_location) as file:
            credentials = yaml_load(file)
        DATABASE_TYPE = str(credentials['DATABASE_TYPE'])
        DBAPI = str(credentials['DBAPI'])
        ENDPOINT = str(credentials['ENDPOINT'])
        USER = str(credentials['DBUSER'])
        PASSWORD = str(credentials['DBPASSWORD'])
        PORT = str(credentials['PORT'])
        DATABASE = str(credentials['DATABASE'])

        self.engine_name = DATABASE_TYPE+"+"+DBAPI+"://" + \
            USER+":"+PASSWORD+"@"+ENDPOINT+":"+PORT+"/"+DATABASE

    def download_table(self, table: str) -> None:
        """
        Downloads the table.

        Attributes:
            table(str): The name of the table.  
        """

        file_location = "data/" + table + "_table.json"

        if isfile(file_location) == True:
            print(file_location, " is already downloaded, skipping")
        else:
            print("Connecting to database...")
            engine = create_engine(self.engine_name)
            df = pd.read_sql_table(table, engine)

            print("Saved as ", file_location)
            df.to_json(file_location)
        return None


if __name__ == "__main__":

    credentials_location = ".gitignore/credentials_for_marketplace.yml"

    downloader = DownloadTables(credentials_location)
    downloader.download_table("products")
