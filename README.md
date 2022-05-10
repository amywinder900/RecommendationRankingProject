# RecommendationRankingProject
Creating a Facebook style marketplace data pipeline. 
# Milestone 1 
The download_products_table.py file retrieves information from a RDS database, securely accressing credentials from a yaml on the local machine.

The image cleaning.py file processes the table from the database. It will:
    - drop unneeded columns
    - remove null values
    - covert the price column to a float
    - create columns for the longitude and latitude
    - encode the main_category as categorical data
    - remove the duplicates.

The data is then sent to data/products_table_clean.json 

The clean_images.py file takes images from a folder uses the PIL library to clean the jpgs ready for use in machine learning. It standardises the size by adding a black border and reduces the number of channels down to RGB. It uses an alive progress bar to document progress and skips over images which have already been cleaned.

# Milestone 2

In tablular_linear_regression.ipynb I implimented some basic machine learning models to predict the price of an item based on the item category and the location of the poster. I used a grid search to tune hyperparameters for a ridge model, lasso model and an elastic net model via sklearn. The best score I got was by using the Ridge model with an alpha of 10. However, this only provided an accuracy rating (R2) of  0.08755583314162077. This makes sense as the data I'm providing the model isn't particularly relevant to the price of a product. 