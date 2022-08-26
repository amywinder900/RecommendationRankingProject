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
 
===========================================================================
Layer (type:depth-idx)                             Param #
===========================================================================
CombinedModel                                      --
├─TextModel: 1-1                                   --
│    └─Sequential: 2-1                             --
│    │    └─Conv1d: 3-1                            38,656
│    │    └─ReLU: 3-2                              --
│    │    └─MaxPool1d: 3-3                         --
│    │    └─Flatten: 3-4                           --
│    │    └─Linear: 3-5                            25,166,080
│    │    └─ReLU: 3-6                              --
│    │    └─Linear: 3-7                            3,341
├─ImageModel: 1-2                                  --
│    └─ResNet: 2-2                                 --
│    │    └─Conv2d: 3-8                            (9,408)
│    │    └─BatchNorm2d: 3-9                       (128)
│    │    └─ReLU: 3-10                             --
│    │    └─MaxPool2d: 3-11                        --
│    │    └─Sequential: 3-12                       --
│    │    │    └─Bottleneck: 4-1                   (75,008)
│    │    │    └─Bottleneck: 4-2                   (70,400)
│    │    │    └─Bottleneck: 4-3                   (70,400)
│    │    └─Sequential: 3-13                       --
│    │    │    └─Bottleneck: 4-4                   (379,392)
│    │    │    └─Bottleneck: 4-5                   280,064
│    │    │    └─Bottleneck: 4-6                   280,064
│    │    │    └─Bottleneck: 4-7                   280,064
│    │    └─Sequential: 3-14                       --
│    │    │    └─Bottleneck: 4-8                   1,512,448
│    │    │    └─Bottleneck: 4-9                   1,117,184
│    │    │    └─Bottleneck: 4-10                  1,117,184
│    │    │    └─Bottleneck: 4-11                  1,117,184
│    │    │    └─Bottleneck: 4-12                  1,117,184
│    │    │    └─Bottleneck: 4-13                  1,117,184
│    │    └─Sequential: 3-15                       --
│    │    │    └─Bottleneck: 4-14                  6,039,552
│    │    │    └─Bottleneck: 4-15                  4,462,592
│    │    │    └─Bottleneck: 4-16                  4,462,592
│    │    └─AdaptiveAvgPool2d: 3-16                --
│    │    └─Sequential: 3-17                       --
│    │    │    └─Linear: 4-17                      1,049,088
│    │    │    └─ReLU: 4-18                        --
│    │    │    └─Dropout: 4-19                     --
│    │    │    └─Linear: 4-20                      131,328
│    │    │    └─ReLU: 4-21                        --
│    │    │    └─Linear: 4-22                      3,341
├─Sequential: 1-3                                  --
│    └─Linear: 2-3                                 351
===========================================================================
Total params: 49,900,217
Trainable params: 49,229,817
Non-trainable params: 670,400
===========================================================================

