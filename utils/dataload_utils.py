import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")


class geo_npz_Dataset_update(Dataset):
    """
    Dataset class for training/testing with methylation data.
    
    This class loads methylation data and target variables from numpy files,
    and filters the data based on provided train/test indices.
    
    Attributes:
        data: Dictionary containing features, targets and additional information
        target_name: Name of the target variable
        additional_list: List of additional metadata fields to include
        key_list: List of sample IDs to use
    """
    
    def __init__(self, x_npy, index_file, if_train, 
                target_name="age",
                additional_list=["sample_id", "tissue"]):
        """
        Initialize the dataset.
        
        Args:
            x_npy (str): Path to numpy file containing methylation data
            index_file (str): Path to pickle file containing train/test indices
            if_train (bool): Whether to use training or test indices
            target_name (str): Name of the target variable (e.g., 'age')
            additional_list (list): List of additional metadata fields to include
        """
        # self.y_pd = pd.read_csv(y_csv)

        self.data = np.load(x_npy, allow_pickle=True)
        self.target_name = target_name
        self.additional_list = additional_list

        # Load train/test indices
        with open(index_file, 'rb') as f:   
            train_index = pickle.load(f)
            test_index = pickle.load(f)
        
        # Filter data based on if_train flag
        if if_train:
            self.key_list = train_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in train_index]
            for test_name in move_list:
                del self.data.item()[test_name]
        else:
            self.key_list = test_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in test_index]
            for test_name in move_list:
                del self.data.item()[test_name]

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.key_list)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (feature, age, additional) where feature is the methylation data,
                  age is the target, and additional is a dictionary of metadata
        """
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"]
        # feature = M2beta_zx(feature)
        # print("#"*30)
        # print(np.std(feature))
        age = self.data.item().get(key_name)["target"]
        additional = self.data.item().get(key_name)["additional"]
        return feature, age, additional


class geo_npz_Dataset_pretrain(Dataset):
    """
    Dataset class for pretraining with methylation data.
    
    This class loads methylation data and target variables from numpy files,
    and filters the data based on provided train/test/pretrain indices.
    
    Attributes:
        data: Dictionary containing features, targets and additional information
        target_name: Name of the target variable
        key_list: List of sample IDs to use
    """
    
    def __init__(self, x_npy, index_file, data_type,
                 target_name="age"):
        """
        Initialize the pretrain dataset.
        
        Args:
            x_npy (str): Path to numpy file containing methylation data
            index_file (str): Path to pickle file containing train/test/pretrain indices
            data_type (str): Type of data to use ('train', 'test', or 'pretrain')
            target_name (str): Name of the target variable (e.g., 'age')
        """
        # self.y_pd = pd.read_csv(y_csv)

        self.data = np.load(x_npy, allow_pickle=True)
        self.target_name = target_name

        # Load train/test/pretrain indices
        with open(index_file, 'rb') as f:
            train_index = pickle.load(f)
            test_index = pickle.load(f)
            pretrain_index = pickle.load(f)

        # Filter data based on data_type
        if data_type == 'train':
            self.key_list = train_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in train_index]
            for test_name in move_list:
                del self.data.item()[test_name]
        elif data_type == 'test':
            self.key_list = test_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in test_index]
            for test_name in move_list:
                del self.data.item()[test_name]
        else:  # pretrain
            self.key_list = pretrain_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in pretrain_index]
            for test_name in move_list:
                del self.data.item()[test_name]

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.key_list)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (feature, age, additional) where feature is the methylation data,
                  age is the target, and additional is a dictionary of metadata
        """
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"]
        # feature = M2beta_zx(feature)
        # print("#"*30)
        # print(np.std(feature))
        age = self.data.item().get(key_name)["target"]
        additional = self.data.item().get(key_name)["additional"]
        return feature, age, additional


class geo_npz_Dataset_inference(Dataset):
    """
    Dataset class for inference with methylation data.
    
    This class loads methylation data from numpy files and prepares it for inference,
    without requiring target variables.
    
    Attributes:
        data: Dictionary containing features and additional information
        target_name: Name of the target variable
        additional_list: List of additional metadata fields to include
        key_list: List of sample IDs to use
    """
    
    def __init__(self, x_npy, index_file, if_train,
                 target_name="age",
                 additional_list=["sample_id", "tissue"]):
        """
        Initialize the inference dataset.
        
        Args:
            x_npy (str): Path to numpy file containing methylation data
            index_file (str): Path to pickle file containing train/test indices
            if_train (bool): Whether to use training or test indices
            target_name (str): Name of the target variable (e.g., 'age')
            additional_list (list): List of additional metadata fields to include
        """
        # self.y_pd = pd.read_csv(y_csv)
        
        self.data = np.load(x_npy, allow_pickle=True)
        self.target_name = target_name
        self.additional_list = additional_list

        # Load train/test indices
        with open(index_file, 'rb') as f:
            train_index = pickle.load(f)
            test_index = pickle.load(f)

        # Filter data based on if_train flag
        if if_train:
            self.key_list = train_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in train_index]
            for test_name in move_list:
                del self.data.item()[test_name]
        else:
            self.key_list = test_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in test_index]
            for test_name in move_list:
                del self.data.item()[test_name]

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.key_list)

    def __getitem__(self, index):
        """
        Get a sample from the dataset for inference.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (feature, additional) where feature is the methylation data
                  and additional is a dictionary of metadata
        """
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"]
        # feature = M2beta_zx(feature)
        # print("#"*30)
        # print(np.std(feature))
        # age = self.data.item().get(key_name)["target"]
        additional = self.data.item().get(key_name)["additional"]
        return feature, additional

