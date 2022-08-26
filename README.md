# RecommendationRankingProject
Creating a Facebook style marketplace data pipeline. 

# Data Handling 
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

# Shallow Algorithms

tabular_linear_regression.ipynb implements basic machine learning models to predict the price of items based on item category and poster location. A grid search has been used to tune hyperparameters for ridge model, lasso model and elatic net model, using sklearn. 

image_classification.ipynb impliments a pipeline which makes use of sklearns logicstic regression to predict the category of an image. 

## Evaluation 
For the tabular linear regression, the best score was achieved using the Ridge model with an alpha of 10, providing a R2 rating of 0.08755583314162077. While this score isn't particularly good, it makes sense considering that the data used isn't particularly relevant to the price of a product. 

For the image classification, the model produces an accuracy of 16 to 17%. This is better than random guessing, which would give an accuracy of around 8% but the task is not within the capablities of the model.

# Neural Networks
Using Neural Networks, this component of the project predicts the category of a product based on the image and description of the product. 

## Loading data 

## Network Structures 
The image model takes advantage of Resnet50 via transfer learning. The final few layers have been replaced to suit this task. The text model consists of convolution layers and linear layers. Finally the combined model combines the two and applies a final linear layer, meaning that the text model and the image model can be trained seperately. The models and their layers are as follows: 


