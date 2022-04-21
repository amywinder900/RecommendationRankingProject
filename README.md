# RecommendationRankingProject
Creating a Facebook style marketplace data pipeline. 
# Milestone 1 
The download_products_table.py file retrieves information from a RDS database, securely accressing credentials from a yaml on the local machine.

The image cleaning.py file processes the table from the database. [TODO: ADD MORE DETAIL HERE]

The clean_images.py file takes images from a folder uses the PIL library to clean the jpgs ready for use in machine learning. It standardises the size by adding a black border and reduces the number of channels down to RGB. It uses an alive progress bar to document progress and skips over images which have already been cleaned.