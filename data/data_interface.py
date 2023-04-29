import importlib
import inspect
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CustomDataModule(pl.LightningDataModule):
    """Custom Data Module class that extends LightningDataModule"""

    def __init__(self, num_workers=0, dataset='', **kwargs):
        """Initialize the Custom Data Module
        
        Args:
        - num_workers (int): number of workers for data loading
        - dataset (str): name of dataset to use
        - kwargs (dict): dictionary of keyword arguments
        
        Raises:
        - ValueError: if the dataset file name or class name is invalid
        """
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.load_data_module()

    def train_dataloader(self):
        """Returns the training DataLoader"""
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Returns the validation DataLoader"""
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        """Returns the test DataLoader"""
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self):
        """Loads the data module"""
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except Exception as e:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}') from e

    def instantiate(self, **other_args):
        """Instantiates the data module"""
        class_args = inspect.signature(self.data_module.__init__).parameters
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)


if __name__ == '__main__':
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='data_image', type=str)
    parser.add_argument('--data_dir', default='/home/pyz/data/COCO', type=str)
    args = parser.parse_args()
    
    # Create an instance of the Custom Data Module class
    data_module = CustomDataModule(**vars(args))
    
    # Call the setup method
    data_module.setup()
