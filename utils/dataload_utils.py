
#%%
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")



class geo_npz_Dataset_update(Dataset):
    def __init__(self, x_npy, index_file, if_train, 
                target_name = "age",
                additional_list = ["sample_id","tissue"]):
        
        """
        read_x, read_y
        """
        # self.y_pd = pd.read_csv(y_csv)

        self.data = np.load(x_npy, allow_pickle = True)
        self.target_name = target_name
        self.additional_list = additional_list

        with open(index_file, 'rb') as f:   
            train_index = pickle.load(f)
            test_index  = pickle.load(f)
        
        if(if_train):
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
        return len( self.key_list )


    def __getitem__(self, index):
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"]
        # feature = M2beta_zx(feature)
        # print("#"*30)
        # print(np.std(feature))
        age = self.data.item().get(key_name)["target"]
        additional = self.data.item().get(key_name)["additional"]
        return feature, age, additional


class geo_npz_Dataset_pretrain(Dataset):
    def __init__(self, x_npy, index_file, data_type,
                 target_name = "age"):

        """
        read_x, read_y
        """
        # self.y_pd = pd.read_csv(y_csv)

        self.data = np.load(x_npy, allow_pickle = True)
        self.target_name = target_name

        with open(index_file, 'rb') as f:
            train_index = pickle.load(f)
            test_index  = pickle.load(f)
            pretrain_index = pickle.load(f)

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
        else:
            self.key_list = pretrain_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in pretrain_index]
            for test_name in move_list:
                del self.data.item()[test_name]


    def __len__(self):
        return len( self.key_list )


    def __getitem__(self, index):
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"]
        # feature = M2beta_zx(feature)
        # print("#"*30)
        # print(np.std(feature))
        age = self.data.item().get(key_name)["target"]
        additional = self.data.item().get(key_name)["additional"]
        return feature, age, additional


class geo_npz_Dataset_inference(Dataset):
    def __init__(self, x_npy, index_file, if_train,
                 target_name = "age",
                 additional_list = ["sample_id","tissue"]):

        """
        read_x, read_y
        """
        # self.y_pd = pd.read_csv(y_csv)

        self.data = np.load(x_npy, allow_pickle = True)
        self.target_name = target_name
        self.additional_list = additional_list

        with open(index_file, 'rb') as f:
            train_index = pickle.load(f)
            test_index  = pickle.load(f)

        if(if_train):
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
        return len( self.key_list )


    def __getitem__(self, index):
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"]
        # feature = M2beta_zx(feature)
        # print("#"*30)
        # print(np.std(feature))
        # age = self.data.item().get(key_name)["target"]
        additional = self.data.item().get(key_name)["additional"]
        return feature, additional

