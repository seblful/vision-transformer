import os
import sys
import pathlib
import argparse
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

from tqdm import tqdm

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for train_sizefolder with data
parser.add_argument("--data_folder", 
                     default='data', 
                     type=str, 
                     help="the name of folder with data")

# Get an arg dataset_key
parser.add_argument("--dataset_key", 
                     default='alessiocorrado99/animals10',
                     type=str,
                     help="the dataset key of Kaggle datasets")

# Get an arg for kaggle_username
parser.add_argument("--kaggle_username", 
                     type=str, 
                     help="the Kaggle username")

# Get an arg for train_sizefolder with datakaggle_api
parser.add_argument("--kaggle_api",
                     type=str, 
                     help="the Kagggle api")

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
DATA_PATH = args.data_folder
DATASET_KEY = args.dataset_key
KAGGLE_USERNAME = args.kaggle_username
KAGGLE_API = args.kaggle_api


def data_downloader(dataset_key=DATASET_KEY,
                    kaggle_username=KAGGLE_USERNAME,
                    kaggle_api=KAGGLE_API):
    '''
    Downloads data from Kaggle, creates new directories,
    and deletes old files
    '''
    data_path = pathlib.Path(DATA_PATH)
    
    # Check if directory 'data' in repository
    if data_path.is_dir() == False: # change to false
        
        
        # Setting Kaggle environment
        try:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
        except TypeError as username_error:
            sys.exit("You should specify Kaggle username: --kaggle_username <your username>")
            
            
        try:
            os.environ['KAGGLE_KEY'] = kaggle_api
        except TypeError as api_error:
            sys.exit("You should specify Kaggle api: --kaggle_api <your api>")

        api = KaggleApi()
        api.authenticate()
        
        # Creating folder for data
        os.mkdir(DATA_PATH)
        print(f"There are no have directory {DATA_PATH}', creating new one")
        
        with tqdm(total=100) as t:
            # Download data from Kaggle 
            print('Downloading data...')
            api.dataset_download_files(dataset_key, path=data_path)
            print('Data has downloaded')
        
        with tqdm(total=100) as t:
            # Getting name of archive
            zip_path = list(data_path.glob('*.zip'))[0]
            # Extracting files from archive
            print('Extracting files from archive...')
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(data_path)
            print('Files has extracted')

            # Deleting zipfile
            try:
                os.remove(zip_path)
            except PermissionError as ex:
                print(f"Couldn't delete zip file for {ex}")

        # Check if data_path doesn't contain additional directories inside and deleting if does
        list_of_dirs_in_data = [file for file in data_path.iterdir() if file.is_dir()]
        print(list_of_dirs_in_data)
        if len(list_of_dirs_in_data) == 1:
            added_dir_path = list_of_dirs_in_data[0]
            
            list_of_dirs_in_added_dir = os.listdir(added_dir_path)
            
            # Looping through each folder in added dir and moving to data_path
            for image_class in list_of_dirs_in_added_dir:
                original_path = added_dir_path / image_class
                print('Orig', original_path)
                
                new_path = data_path / image_class
                print('new_path', new_path)
                # Moving files
                shutil.move(original_path, new_path)
            print('Folders has removed')
            
            # Deleting emptyExceptionded folder
            try:
                os.remove(added_dir_path)
            except PermissionError as ex:
                print(f"Couldn't delete added dir path for {ex}")
        
        # Checking presence of translate file to rename folders and renaming them
        translate_file_path = data_path / 'translate.py'
        if translate_file_path.is_file():
            # Importing dict with translation
            from data.translate import translate
            # Adding missing key
            translate['ragno'] = 'spider'
            translate['raw-img'] = 'raw-img'
            translate['__pycache__'] = '__pycache__'
            
            # Creating list with folders
            list_of_dirs_in_data = [file for file in data_path.iterdir() if file.is_dir()]
            # Looping through folders in data_path and rename folders
            for folder in list_of_dirs_in_data:
                folder.rename(data_path / translate[folder.name])
                print('Name of folders has renamed')
                
            try:
                os.remove(data_path/'translate.py')
            except PermissionError as ex:
                print(f"Couldn't delete translate.py for {ex}")

    else:
        print("The repository has directory 'data'")

if __name__ == '__main__':
    data_downloader()
