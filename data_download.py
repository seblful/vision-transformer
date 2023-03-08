
import os
import pathlib
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

from tqdm import tqdm

def data_downloader(dataset_key='alessiocorrado99/animals10',
                    kaggle_username='seblful',
                    kaggle_api='6757c11d0dc1367e16a1a6b9fab157d1'):
    '''
    Downloads data from Kaggle, creates new directories,
    and deletes old files
    '''
    data_path = pathlib.Path('data')
    
    # Check if directory 'data' in repository
    if data_path.is_dir() == False: # change to false
        os.mkdir('data')
        print("There are no have directory 'data', creating new one")
        
        # Setting Kaggle environment
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_api

        api = KaggleApi()
        api.authenticate()
        
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

data_downloader()
