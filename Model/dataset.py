
from torch.utils.data.dataset import Dataset
import os 
from PIL import Image
import torch
import numpy as np 

class DiffusionDataset(Dataset) : 
    
    def __init__(self ,data_path : str ,input_shape : tuple) :
        self.input_shape = input_shape 
        self.data_path = data_path
        
        self.images_files = os.listdir(path=data_path)
        
        
    def __len__(self) : 
        return len(self.images_files) 
    
    def __getitem__(self, index) :
        image = Image.open(self.images_files)
        
        return image
    
def Diffusion_dataset_collate(batch):
    images = []
    for image in batch:
        images.append(image)
    images = torch.from_numpy(np.array(images, np.float32))
    return images