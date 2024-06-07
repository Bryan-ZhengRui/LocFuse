from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pickle
import numpy as np
import cv2
import torch
import random
import yaml
import random

config_filename = os.path.join('configs/config.yaml')
config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
train_path = config['Robotcar']['train_pickle_path_raw']
test_query_path = config['Robotcar']['test_query_pickle_path']
test_database_path = config['Robotcar']['test_database_pickle_path']
    

class Robotcar_Dataset_qua(Dataset):
    """A custom dataset that loads RGB and BEV images as inputs."""

    def __init__(self, transform=None):
        """
        Args:
            root_dir (string): Directory containing RGB and BEV images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(Robotcar_Dataset_qua, self).__init__()
        self.rootpath = os.path.abspath(__file__)
        self.transform = transform
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.query_filepath = os.path.join(str(dir_path), os.pardir, train_path)
        with open(self.query_filepath, 'rb') as file:
            self.queries = pickle.load(file)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):

        # Load RGB and BEV images
        result = {'idx': idx}

        # Load rgb anchor
        rgb_pathname = os.path.join(self.queries[idx]['query_rgb'])
        img_rgb = load_png(rgb_pathname)       
        img_rgb = torch.tensor(img_rgb, dtype=torch.float32)
        
        # Load BEV anchor
        bev_pathname = os.path.join(self.queries[idx]['query_bev'])
        img_bev = load_png(bev_pathname, transform=self.transform)
        img_bev = torch.tensor(img_bev, dtype=torch.float32)

        #load postives and negtives
        img_rgb_p, img_bev_p = self.get_positive_pair(idx)
        img_rgb_n, img_bev_n = self.get_negative_pair(idx)
        img_rgb_n_nearst, img_bev_n_nearst = self.get_nearst_negative_pair(idx)

        result['anchor'] = {
            'rgb': (img_rgb),
            'bev': (img_bev),}
        
        result['positives'] = {
            'rgb': (img_rgb_p),
            'bev': (img_bev_p),}
    
        result['negatives'] = {
            'rgb': (img_rgb_n),
            'bev': (img_bev_n),}
        
        result['negatives_nearst'] = {
            'rgb': (img_rgb_n_nearst),
            'bev': (img_bev_n_nearst),}
        
        return result
    
    def get_positive_pair(self, idx):
        if len(self.queries[idx]['positives'])>10:
            index_p = random.choice(self.queries[idx]['positives'][-10:])
        else:
            index_p = random.choice(self.queries[idx]['positives'])
        # index_p = random.choice(self.queries[idx]['positives'])
        rgb_path = self.queries[index_p]['query_rgb']
        bev_path = self.queries[index_p]['query_bev']
        img_rgb_p = load_png(rgb_path)
        img_bev_p = load_png(bev_path)

        return img_rgb_p, img_bev_p
    
    def get_negative_pair(self, idx):
        # if len(self.queries[idx]['negatives'])>250:
        #     index_n = random.choice(self.queries[idx]['negatives'][0:250])
        # else:
        #     index_n = random.choice(self.queries[idx]['negatives'])
        index_n = random.choice(self.queries[idx]['negatives'])
        rgb_path = self.queries[index_n]['query_rgb']
        bev_path = self.queries[index_n]['query_bev']
        img_rgb_n = load_png(rgb_path)
        img_bev_n = load_png(bev_path)
        return img_rgb_n, img_bev_n     
    
    def get_nearst_negative_pair(self, idx):
        if len(self.queries[idx]['negatives'])>20:
            index_n = random.choice(self.queries[idx]['negatives'][0:20])
        else:
            index_n = random.choice(self.queries[idx]['negatives'])
        rgb_path = self.queries[index_n]['query_rgb']
        bev_path = self.queries[index_n]['query_bev']
        img_rgb_n = load_png(rgb_path)
        img_bev_n = load_png(bev_path)
        return img_rgb_n, img_bev_n     
    
    
    
    


class TEST_Robotcar_Dataset(Dataset):
    """A custom dataset that loads RGB and BEV images as inputs."""

    def __init__(self, transform=None):
        """
        Args:
            root_dir (string): Directory containing RGB and BEV images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(TEST_Robotcar_Dataset, self).__init__()
        self.rootpath = os.path.abspath(__file__)
        self.transform = transform
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.query_filepath = os.path.join(str(dir_path), os.pardir, test_query_path)
        with open(self.query_filepath, 'rb') as file:
            self.queries = pickle.load(file)
        self.num_of_each_run = [len(self.queries[i]) for i in range(len(self.queries))]

    def __len__(self):
        return sum(self.num_of_each_run)

    def __getitem__(self, idx):
        # Load RGB and BEV images    
        [idx1, idx2] = getidx2test(idx, self.num_of_each_run)
        result = {'seq_idx': int(idx1)}
        # Load rgb
        rgb_pathname = os.path.join(os.pardir,os.pardir,'common/datasets/Oxford_RobotCar_raw/',self.queries[idx1][idx2]['query_rgb'])
        img_rgb = load_png(rgb_pathname)         
        img_rgb = torch.tensor(img_rgb, dtype=torch.float32) 
        # Load BEV
        bev_pathname = os.path.join(os.pardir,os.pardir,'common/datasets/Oxford_RobotCar_raw/', self.queries[idx1][idx2]['query_bev'])
        img_bev = load_png(bev_pathname)
        img_bev = torch.tensor(img_bev, dtype=torch.float32)
        result['query'] = {
            'frame_idx': (int(idx2)),
            'rgb': (img_rgb),
            'bev': (img_bev),}

        return result
    
       
class Data_Robotcar_Dataset(Dataset):
    """A custom dataset that loads RGB and BEV images as inputs."""

    def __init__(self, transform=None):
        """
        Args:
            root_dir (string): Directory containing RGB and BEV images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(Data_Robotcar_Dataset, self).__init__()
        self.rootpath = os.path.abspath(__file__)
        self.transform = transform
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.database_filepath = os.path.join(str(dir_path), os.pardir, test_database_path)
        with open(self.database_filepath, 'rb') as file:
            self.database = pickle.load(file)
        self.num_of_each_run = [len(self.database[i]) for i in range(len(self.database))]

    def __len__(self):
        return sum(self.num_of_each_run)

    def __getitem__(self, idx):
        # Load RGB and BEV images    
        [idx1, idx2] = getidx2test(idx, self.num_of_each_run)
        result = {'seq_idx': int(idx1)}
        # Load rgb
        rgb_pathname = os.path.join(os.pardir,os.pardir,'common/datasets/Oxford_RobotCar_raw/',
                                    self.database[idx1][idx2]['database_rgb'])
        img_rgb = load_png(rgb_pathname)         
        img_rgb = torch.tensor(img_rgb, dtype=torch.float32) 
        # Load BEV
        bev_pathname = os.path.join(os.pardir,os.pardir,'common/datasets/Oxford_RobotCar_raw/', 
                                    self.database[idx1][idx2]['database_bev'])
        img_bev = load_png(bev_pathname)
        img_bev = torch.tensor(img_bev, dtype=torch.float32)
        result['query'] = {
            'frame_idx': (int(idx2)),
            'rgb': (img_rgb),
            'bev': (img_bev),}

        return result
    


    

def load_png(path, transform = None):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if transform is not None:
        ran = random.random()
        if ran < 0.15:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif ran < 0.25:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif ran < 0.35:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = np.float32(img) / 255
    img = np.transpose(img, [2, 0, 1])
    return img
    

def getidx2test(idx, num_of_each_run=[]):
    sum=-1
    for idx1 in range(len(num_of_each_run)):    
        if idx > sum and idx <= sum+num_of_each_run[idx1]:
            idx2 = idx - sum - 1
            return [idx1, idx2]
        sum += num_of_each_run[idx1]
    
    return idx1





        
if __name__ == '__main__':
    os.chdir(os.pardir)
    my_dataset = TEST_Robotcar_Dataset()
    print(my_dataset[0])
    print(len(my_dataset))
