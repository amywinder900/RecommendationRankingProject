from yaml import safe_load as yaml_load
from sqlalchemy import create_engine
import pandas as pd

with open(".gitignore/credentials_for_marketplace.yml") as file:
    credentials  = yaml_load(file)
DATABASE_TYPE = credentials['DATABASE_TYPE']
DBAPI = credentials['DBAPI'] 
ENDPOINT = credentials['ENDPOINT']       
USER = credentials['DBUSER']
PASSWORD = credentials['DBPASSWORD']
PORT = credentials['PORT']
DATABASE = credentials['DATABASE']

print("Connecting to database...")
engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")        
df = pd.read_sql_table("products", engine)     

print("Saved as products_table.json")
df.to_json("products_table.json")