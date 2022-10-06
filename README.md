# RecommendationRankingProject
Creating a Facebook style marketplace data pipeline. 

# Data Handling 
This module processes data and stores locally in a /data folder. 

## download_products_table.DownloadTables
The download_products_table.py file retrieves information from a RDS database, securely accressing credentials from a yaml on the local machine. The download will not be attempted if the table already exists on the local machine. 

    Attributes:
        crediential_location(str): The location of the yaml file containing the database credentials. It should contain DATABASE_TYPE, DBAPI, ENDPOINT, USER, PASSWORD, PORT, DATABASE

    Methods:
        download_table: Downloads the table.
            Attributes:
                table(str): The name of the table.  

### How to use

First instantiate the class.
> downloader = DownloadTables(credentials.yml)

Then download the table as a json file using the download_table method. 
> downloader.download_table("Sales")

## clean_tabular
The image cleaning.py file processes the table from the database. Running the file will:
    - remove null values
    - covert the price column to a float
    - create columns for the longitude and latitude
    - encode the main_category as categorical data
    - remove the duplicates
The data is then sent to data/products_table_clean.json 

## clean_tabular.CleanTabular
This class is used to clean tabular data.

    Attributes:
        product_df (DataFrame): the dataframe.

    Methods:
        drop_columns: Takes in a list of columns and drops them from the dataframe.
            Attributes:
                columns(list): The list of column names.

        remove_null: Removes the rows with null values. 

        clean_prices:Cleans the prices and converts to integer. 

        retrieve_longitude_and_latitude: A static method. Given a location of the form "city, district" or similar, retrieves the longitude and latitude.

        create_longitude_and_latitude_columns: Adds columns for longitude and latitude in the DataFrame. 

        create_main_category_column: Creates a column which displays the main category. 

        encode_categorical_data: Given the name of a column, this method encodes the categorical data by creating a new column for each category with binary True/False values. 

        remove_duplicates: Removes the duplicates from the dataframe. 

        prepare_data: Method to run the standard data cleaning steps so that it is ready for basic machine learning algorithms. 

        get_product_df: Retrieves the dataframe. 

## clean_images.py

The clean_images.py file takes images from a folder uses the PIL library to clean the jpgs ready for use in machine learning. It standardises the size by adding a black border and reduces the number of channels down to RGB. It uses an alive progress bar to document progress and skips over images which have already been cleaned.

## clean_images.CleanImage
This class is used to clean a folder of images. It will create a folder containing the new images. 

    Attributes:
        folder(str): The path to the folder containing the images in jpg format. Should be of the format "folder/to/images"

        target_folder(str): The target folder to send the cleaned images to.

        final_image_size(int): The dimensions for the final images. Default is 512.

    Methods:
    retrieve_images: Collects a list of the images from the folder.

    clean_image: Collects a list of the images from the folder.
    
    clean_all_images: Retrives the jpg images from the folder and cleans them.

# Shallow Algorithms

tabular_linear_regression.ipynb implements basic machine learning models to predict the price of items based on item category and poster location. A grid search has been used to tune hyperparameters for ridge model, lasso model and elatic net model, using sklearn. 

image_classification.ipynb impliments a pipeline which makes use of sklearns logicstic regression to predict the category of an image. 

## Evaluation 
For the tabular linear regression, the best score was achieved using the Ridge model with an alpha of 10, providing a R2 rating of 0.08755583314162077. While this score isn't particularly good, it makes sense considering that the data used isn't particularly relevant to the price of a product. 

For the image classification, the model produces an accuracy of 16 to 17%. This is better than random guessing, which would give an accuracy of around 8% but the task is not within the capablities of the model.

# Neural Networks
Using Neural Networks, this component of the project predicts the category of a product based on the image and description of the product. 

## Dataloaders
Data for the neural networks have been created using torch.utils.data.Dataset. 

### image_loader.ProductImageCategoryDataset

The ProductImageCategoryDataset object inherits its methods from the torch.utils.data.Dataset module.
It loads images from a numpy array. The images should be processed so that the height and width is the same. 

    Parameters:

        images_location(str): The folder location containing the cleaned images.

        img_side_length(int): The side length of the images. 

        load_image_category_table(bool): If true, the class loads the image category table from data/image_category_table.json or if False it will create a new one. Default value is False.

        transform: The transformation or list of transformations to be done to the image. If no transform is passed, the class will do a generic transformation to resize, convert it to a tensor, and normalize the numbers.

        decoder(dict): The dictionary which assigns each category to a number, with the key being the number and the item being the category name. Default is None.

### image_loader.create_data_loaders

This function creates the dataloaders with a training and validation split. 

    Parameters:

      images_location(str): The path to the file containing the cleaned images. 

      image_dataset(Dataset): The dataset object which prepares the image and the category label. 

      image_side_length(int): The side length of the images. 

      image_transforms(dict): A dictionary containing the the transformation or list of transformations to be done to the image. It should contain transformations for the train and validation phases using the keys "train" and "val". 

      validation_split(float): The proportion of the dataset which should be used for validation. Default is 0.2

      batch_size(int): The batch size to process the images in. 

      shuffle(bool): Whether or not to shuffle the order of the images. Default is true.

    Returns:

        data_loader(dict): The dictionary of dataloaders with keys "train" and "val". 

        dataset_sizes(dict): The dictionary of dataset sizes with keys "train" and "val".

### text_loader.ProductTextCategoryDataset
The ProductTextCategoryDataset object inherits its methods from the torch.utils.data.Dataset module.

    Parameters:

        categories(): The pandas series corresponding to the product category labels.

        descriptions(): The pandas series corresponding to the product descriptions. This should be in the same order as the category labels.

        max_length(int): The number of words to consider in the description. Default is the first 50.

        decoder(dict): The dictionary which assigns each category to a number, with the key being the number and the item being the category name. Default is None.

### text_loader.create_data_loaders
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

### combined_loader.CombinedLoader

The ImageTextProductDataset object inherits its methods from the torch.utils.data.Dataset module.

    Parameters:

        images_location(str): The path to the file containing the cleaned images. 

        image_side_length(int): The side length of the images. 

        load_image_category_table(bool): If True, loads the combined image and description table. Otherwise downloads the tables from the database. Default is False.

        image_transforms(dict): A dictionary containing the the transformation or list of transformations to be done to the image. It should contain transformations for the train and validation phases using the keys "train" and "val". Default is None. 

        max_text_length(int): The number of words to consider in the description. Default is the first 50.

        decoder(dict): The dictionary which assigns each category to a number, with the key being the number and the item being the category name. Default is None.

### combined_loader.create_data_loaders
This function creates the dataloaders with a training and validation split. 

    Parameters:

      images_location(str): The path to the file containing the cleaned images. 

      image_dataset(Dataset): The dataset object which prepares the image and the category label. 

      image_side_length(int): The side length of the images. 

      image_transforms(dict): A dictionary containing the the transformation or list of transformations to be done to the image. It should contain transformations for the train and validation phases using the keys "train" and "val". 

      validation_split(float): The proportion of the dataset which should be used for validation. Default is 0.2

      batch_size(int): The batch size to process the images in. 

      shuffle(bool): Whether or not to shuffle the order of the images. Default is true.



## Network Structures 
The image model takes advantage of Resnet50 via transfer learning. The final few layers have been replaced to suit this task. The text model consists of convolution layers and linear layers. Finally the combined model combines the two and applies a final linear layer, meaning that the text model and the image model can be trained seperately. The models and their layers are as follows:
 
Layer (type:depth-idx)                             Param #

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

Total params: 49,900,217
Trainable params: 49,229,817
Non-trainable params: 670,400


# API
To allow a user to predict product category without the need for coding ability, FastAPI is utilised. 

This can be run using uvicorn to update the localhost to run the api.py file. 

## data_processor
The ImageProcessor and TextProcessor classes prepare data to be sent in to the models. ImageProcessor can be called on a PIL Image. TextProcessor can be called on a string.

## Cloud Migration

A docker-compose file allows for real time updates of the docker image, as well as potential application expansion further down the line. To allow the code to function as expected, a combined_model_state.pt file is required in the dockerfolder.
