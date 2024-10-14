import torch
# import pandas as pd
from PIL import Image
# import os
import torchvision.transforms as TF 
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np


def get_compose_list(img_size):
    '''
    get the compose function list based on img size
    '''
    assert len(img_size) == 3

    compose_list=[]

    if img_size[0] == 1:
        compose_list.append(TF.Grayscale(num_output_channels=img_size[0]))

    compose_list.extend(
        # every setting has the followings
        [TF.ToTensor(),
        TF.Resize((img_size[1],img_size[2]), 
                interpolation=TF.InterpolationMode.BICUBIC, 
                antialias=True),
        TF.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1] 
        ]
    )
    return compose_list



class ChestXrayDataset(Dataset):
    def __init__(self, img_data_dir, 
                 ds_name,
                 df_data, 
                 img_size=(1,224,224),
                 augmentation=False, 
                 label='Edema',
                 sensitive_label= 'sex',
                 ):
        self.img_data_dir = img_data_dir
        self.ds_name = ds_name  
        self.df_data = df_data
        self.img_size = img_size
        self.do_augment = augmentation
        self.label = label
        self.sensitive_label = sensitive_label

        if self.sensitive_label == 'sex':
            self.col_name_a = 'sex_label'

        else:
            print('{} not implemented'.format(self.sensitive_label))
            raise NotImplementedError



        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])
        self.transform = TF.Compose(get_compose_list(self.img_size))

        self.samples = []

        for idx in tqdm((self.df_data.index), desc='Loading Data'):
            if self.ds_name == 'chexpert':
                col_name_pth = 'path_preproc'
            elif self.ds_name == 'NIH':
                col_name_pth = 'Image Index'
                
            path_preproc_idx = self.df_data.columns.get_loc(col_name_pth)
            img_path = self.img_data_dir + self.df_data.iloc[idx, path_preproc_idx]
            img_label = np.array(self.df_data.loc[idx, self.label.strip()] == 1, dtype='float32')
            sensitive_attribute = np.array(self.df_data.loc[idx, self.col_name_a.strip()] == 1, dtype='float32')
            sample = {'image_path': img_path, 'label': img_label, 'sensitive_attribute': sensitive_attribute}
            self.samples.append(sample)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        sample = self.get_sample(item)
        # image = T.ToTensor()(sample['image'])
        image = sample['image']
        label = torch.from_numpy(sample['label'])
        sensitive_attribute = torch.from_numpy(sample['sensitive_attribute'])
        


        if self.do_augment:
            image = self.augment(image)

        return {'image': image, 'label': label, 'sensitive_attribute': sensitive_attribute}

    def get_sample(self, item):
        sample = self.samples[item]

        image = Image.open(sample['image_path']).convert('RGB') #PIL image
        image = self.transform(image)
    

        return {'image': image, 'label': sample['label'], 'sensitive_attribute': sample['sensitive_attribute']}

    def exam_augmentation(self,item):
        assert self.do_augment == True, 'No need for non-augmentation experiments'

        sample = self.get_sample(item) #PIL
        image = T.ToTensor()(sample['image'])

        if self.do_augment:
            image_aug = self.augment(image)

        image_all = torch.cat((image,image_aug),axis= 1)
        assert image_all.shape[1]==self.image_size[0]*2, 'image_all.shape[1] = {}'.format(image_all.shape[1])
        return image_all

def reduce_dataset(dataset, num_samples, seed=42):
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # reduce the dataset to num_samples by randomly selecting samples
    crit = True
    while crit:
        indices = np.random.choice(len(dataset), num_samples, replace=False,)
        dataset_reduced = torch.utils.data.Subset(dataset, indices)
        reduced_dataloader = DataLoader(dataset_reduced, batch_size=32, shuffle=False)
        '''
       
        labels = [dataset_reduced.samples[i]['label'] for i in range(len(dataset_reduced))]
        sensitive = [dataset_reduced.samples[i]['sensitive_attribute'] for i in range(len(dataset_reduced))]

        if labels.count(0) > 0 and labels.count(1) > 0 and sensitive.count(0) > 0 and sensitive.count(1) > 0:
            crit = False
            


        '''
        # check if minimum number of samples are present for each class
        
        # get all labels and sensitive attributes
        label_count_0 = 0
        label_count_1 = 0
        sensitive_count_0 = 0
        sensitive_count_1 = 0
        for data in tqdm(reduced_dataloader):
            labels = data['label'].cpu().numpy().tolist()
            sensitive_attributes = data['sensitive_attribute'].cpu().numpy().tolist()
            label_count_0 += labels.count(0)
            label_count_1 += labels.count(1)
            sensitive_count_0 += sensitive_attributes.count(0)
            sensitive_count_1 += sensitive_attributes.count(1)

            if label_count_0 > 0 and label_count_1 > 0 and sensitive_count_0 > 0 and sensitive_count_1 > 0:
                crit = False
                break
        print('label_count_0:', label_count_0, 'label_count_1:', label_count_1, 'sensitive_count_0:', sensitive_count_0, 'sensitive_count_1:', sensitive_count_1)
        
    return dataset_reduced

def distort_data(dataset, distort_prc, seed=42):
    from copy import deepcopy
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_distort = deepcopy(dataset)

    samples = dataset_distort.samples

    # get male samples
    idx_male_healthy = [sample.idx for sample in samples if sample['sensitive_attribute'] == 1 and sample['label'] == 0]
    N_male_healthy = len(idx_male_healthy)

    # Distort the labels
    idx_flip =  np.random.choice(N_male_healthy, int(N_male_healthy*distort_prc), replace=False,)

    for idx in idx_flip:
        samples[idx_male_healthy[idx]]['label'] = np.array(1., dtype=np.float32)

    # put samples into test_dataset_distort
    dataset_distort.samples = samples

    return dataset_distort

