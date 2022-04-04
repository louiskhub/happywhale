# Happywhale

As data source, we use the Dataset from the "Happywhale - Whale and Dolphin Identification" competition from [Kaggle](https://www.kaggle.com/competitions/happy-whale-and-dolphin/data). The goal is to identify new images from whales and assigning them to existing or identifying them as new whales. 



## Virtual Environment

In order to install all necessary dependencies, please make sure you have python3 and a local [Conda](https://docs.conda.io/en/latest/) distribution (e.g. anaconda or miniconda) installed.

After you cloned this repo, you can install the necessary dependencies via `pip install -r requirements.txt`.



## Getting the Data

You can download the original data from [Kaggle](https://www.kaggle.com/competitions/happy-whale-and-dolphin/data) but you have to make sure that you have enough free space on your disk.
After that you can run `crop_and_save_images.py` to downsize the images. Please make sure that you name the directory that you downloaded from Kaggle like this:`KaggleData`.

If you want to save time, you can also download our already preprocessed images and csv files from [Google Drive](https://drive.google.com/drive/folders/1SN__44h9bDxHrDB94WSeSGEDaxx3ISyo). Everything you need is stored in `OurTrainingData.zip`. Make sure to unzip the file in the same parent directory as this repository.

You can also find useful logs and nice visualisationsfrom our training process on [Google Drive](https://drive.google.com/drive/folders/1SN__44h9bDxHrDB94WSeSGEDaxx3ISyo)!



## Filestructure

src - python directory containing all of our main code

visualizations - containing visualisation & data analysis notebooks of the training dataset and really nice visaulisation of our training process, have a look at them!

training - containing two examplitory training pipeline notebooks
