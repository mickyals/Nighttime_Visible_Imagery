#imports
import os
import numpy as np
import pandas as pd
from siphon.catalog import TDSCatalog
import xarray as xr
import warnings
import torch
#from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")



class ThreddsData():
    def __init__(self, timeframe, catalog_url, storage_path=None, openDAP_url_regex ='https://redoak.cs.toronto.edu/twitcher/ows/proxy/thredds/dodsC/datasets/GOES17/ABI-L2-MCMIPM/'):
        self.years = timeframe[0]
        self.days_of_year = timeframe[1]
        self.hour_of_day = timeframe[2]
        self.catalog = catalog_url
        self.openDAP = openDAP_url_regex
        
        # Create the storage path
        # Set storage_path to the current working directory if not provided
        self.storage_path = storage_path if storage_path is not None else os.path.join(os.getcwd(), 'Data')
        os.makedirs(self.storage_path, exist_ok=True)  


    def get_data(self):

        base_url = 'https://redoak.cs.toronto.edu/twitcher/ows/proxy/thredds/catalog/datasets/GOES17/ABI-L2-MCMIPM/'
        images_path = os.path.join(self.storage_path, 'goes_images')
        metadata_path = os.path.join(self.storage_path, 'metadata')
        OpenDAP_urls = []
        npy_total = 0
        meta_total = 0


        for year in self.years:
            for day in self.days_of_year:
                for hour in self.hour_of_day:
                    day_str = str(day).zfill(3)
                    url = f'{base_url}{year }/{day_str}/{hour}/catalog.html'

                    try:
                        catalog = TDSCatalog(url)
                        datasets = [file for file in catalog.datasets if 'MCMIPM1' in str(file)]
                        datasets = sorted(datasets)
                        print(f'Found {len(datasets)} datasets for {year} on day {day_str} at hour {hour}')
                    except Exception as e:
                        print(f'Error loading catalog {e}')
                        continue

                    try:
                        os.makedirs(images_path, exist_ok=True)
                        os.makedirs(metadata_path, exist_ok=True)


                        # get images
                        dataset_name = datasets[0]
                        url = f'{self.openDAP}{year}/{day_str}/{hour}/{dataset_name}'
                        x_dataset = xr.open_dataset(url)
                        npy = self.process_xarray(x_dataset)
                        name = f'MCMIPM1_{year}_{day_str}_{hour}_{dataset_name[-8:-3]}'
                        np.save(os.path.join(images_path, f'{name}.npy'), npy)

                        npy_total += 1


                        # get metadata
                        meta = x_dataset['CMI_C01']
                        meta.to_netcdf(os.path.join(metadata_path, f'{name}'))
                        print(f'Successfully downloaded dataset {dataset_name[:-3]}')

                        meta_total += 1

                        OpenDAP_urls.append(url)

                    except Exception as e:
                        print(f'Error loading dataset: {e}')
                        continue
        df = {'openDAP URL': OpenDAP_urls}
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(self.storage_path, 'openDAP_urls.csv'), index=True)

        print(f'Done! A total of {npy_total} were created and a total of {meta_total} were saved.')    
                        
                               

    def process_xarray(self, xarray_dataset, variable_list=None):
        if variable_list is None:
            variable_list = ['CMI_C01','CMI_C02','CMI_C03','CMI_C04','CMI_C05',
                                'CMI_C06','CMI_C07','CMI_C08','CMI_C09','CMI_C10',
                                'CMI_C11','CMI_C12','CMI_C13','CMI_C14','CMI_C15','CMI_C16']
            

        try:
            selected_data = [xarray_dataset[variable] for variable in variable_list]
            green_band = 0.48358168 * xarray_dataset['CMI_C02'] + 0.45706946 * xarray_dataset['CMI_C01'] + 0.06038137 * xarray_dataset['CMI_C03']
            selected_data.append(green_band)
            variable_array = np.stack(selected_data, axis=0)
            assert variable_array.shape == (17, 500, 500)

            return variable_array

        except Exception as e:
            print(f'Error processing data: {e}')
            return None



class GoesNumpyDataset(Dataset):
    """
    A PyTorch Dataset class for loading numpy files containing GOES satellite data.

    Args:
        data_dir (str): The directory containing the numpy files.
        x_idxs (list): The indices of the input variables.
        y_idxs (list): The indices of the output variables. If None, output is None.
        transform (callable): The transformation to apply to the input data.
        vis_transform (callable): The transformation to apply to the output data.
    """
    def __init__(self, data_dir, x_idxs, y_idxs, transform=None):
        self.data_dir = data_dir
        self.x_idxs = x_idxs
        self.y_idxs = y_idxs
        self.transform = transform
        self.files = [file for file in os.listdir(data_dir) if file.endswith('.npy')]
        

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.files)


    def __getitem__(self, index):
        """
        Returns the input and output data for the given index.

        Args:
            index (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the input data and the output data.
        """
        # Get the file path of the data
        file_path = os.path.join(self.data_dir, self.files[index])

        # Load the data from the file
        data = np.load(file_path)

        # Convert the data to a PyTorch tensor
        data = torch.from_numpy(data).float()

        # Extract the input and output data based on the given indices
        input_x = data[self.x_idxs, :, :]
        if len(input_x.shape) == 2:
            input_x = input_x.unsqueeze(0)

        if self.y_idxs is None:
            # Apply the transform to the input data
            input_x = self.transform(input_x)
            # Return the input data
            return input_x

        else:
            target_y = data[self.y_idxs, :, :]
            if len(target_y.shape) == 2:
                target_y = target_y.unsqueeze(0)

            target_y = np.clip(target_y, 0, 1)
            target_y = self.transform(target_y)
            # Apply the transform to the input data
            input_x = self.transform(input_x)

            # Return the input and target data
            return input_x, target_y


def get_dataloader(dataset: Dataset, batch_size: int, seed: int, shuffle: bool=True, validation_split: float=0.2):

    if validation_split > 0.0:
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)), test_size=validation_split, shuffle=shuffle, random_state=seed)
        
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader
    
if __name__ == "__main__":
    years = [2018, 2019, 2020, 2021]
    days = np.arange(1,365, 5)
    hours = [18, 19, 20, 21]
    catalog_url = 'https://redoak.cs.toronto.edu/twitcher/ows/proxy/thredds/catalog/datasets/GOES17/ABI-L2-MCMIPM/catalog.xml'
    timeframe = [years, days, hours]
    thredds = ThreddsData(timeframe, catalog_url)
    thredds.get_data()
        