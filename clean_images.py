from PIL import Image
from os import makedirs
from alive_progress import alive_bar
import glob


class CleanImages:
    """
    This class is used to clean a folder of images. It will create a folder containing the new images. 

    Attributes:
        folder(str): The path to the folder containing the images in jpg format. Should be of the format "folder/to/images"
        target_folder(str): The target folder to send the cleaned images to. 
        final_image_size(int): The dimensions for the final images. Default is 512.
    """

    def __init__(self, folder: str, target_folder: str, final_image_size: int = 512) -> None:
        """
        See help(CleanImages)
        """
        self.folder = folder
        self.target_folder = target_folder
        self.final_image_size = final_image_size
        makedirs(target_folder, exist_ok=True)

    def retrieve_images(self, folder_name: str) -> list:
        """
        Collects a list of the images from the folder.

        Attributes:
            folder_name(str): The name of the folder to search.

        Returns:

            list_of_images(str): The list of jpg images in the folder.
        """
        search_term = folder_name + "*.jpg"
        list_of_images = glob.glob(search_term)
        return list_of_images

    def clean_image(self, image_name: str) -> Image:
        """
        Cleans an image by resizing, pasting onto a square background and converting to RGB. 

        Attributes:
            image_name(str): The name of the image to be cleaned. 

        Returns: 
            final_image(Image): The cleaned image.
        """

        # open image
        img = Image.open(image_name)

        # resize by finding the biggest side of the image and calculating ratio to resize by
        max_side_length = max(img.size)
        resize_ratio = self.final_image_size / max_side_length
        img_width = int(img.size[0]*resize_ratio)
        img_height = int(img.size[1]*resize_ratio)
        img = img.resize((img_width, img_height))

        # convert to rgb
        img = img.convert("RGB")

        # paste on black image
        final_img = Image.new(mode="RGB", size=(
            self.final_image_size, self.final_image_size))
        final_img.paste(img, ((self.final_image_size - img_width) //
                        2, (self.final_image_size - img_height)//2))

        return final_img

    def clean_all_images(self) -> None:
        """
        Retrives the jpg images from the folder and cleans them.
        """
        path_to_images = self.retrieve_images(self.folder)
        path_to_cleaned_images = self.retrieve_images(self.target_folder)

        list_of_cleaned_images = [x.split("/")[-1]
                                  for x in path_to_cleaned_images]

        with alive_bar(len(path_to_images)) as bar:            
            for image_path in path_to_images:
                image_name = image_path.split("/")[-1]
                
                if image_name not in list_of_cleaned_images:
                    #print("Cleaning ", image_name)
                    img = self.clean_image(image_path)
                    # save to file
                    new_file_path = self.target_folder + image_name
                    img.save(new_file_path)
                #else:
                    #print(image_name, " is already in target folder.")
                
                bar()

        return None


if __name__ == "__main__":
    cleaner = CleanImages("data/images/", "data/cleaned_images/")
    cleaner.clean_all_images()
