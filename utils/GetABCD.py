from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils
import torch.utils.data as Data
import torch

def dataset_factory(cfg: DictConfig, bs: int):

    datasets = load_abcd_data(cfg)

    dataloaders = init_stratified_dataloader(cfg, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)
    
    x_train = dataloaders[0].dataset.dataset.tensors[0]
    y_train = dataloaders[0].dataset.dataset.tensors[2]
    x_valid = dataloaders[1].dataset.dataset.tensors[0]
    y_valid = dataloaders[1].dataset.dataset.tensors[2]
    x_test = dataloaders[2].dataset.dataset.tensors[0]
    y_test = dataloaders[2].dataset.dataset.tensors[2]

    x_train = torch.Tensor(x_train).unsqueeze(1)
    x_valid = torch.Tensor(x_valid).unsqueeze(1)
    x_test = torch.Tensor(x_test).unsqueeze(1)

    train_dataset = Data.TensorDataset(x_train, y_train[:,0])
    valid_dataset = Data.TensorDataset(x_valid, y_valid[:,0])
    test_dataset = Data.TensorDataset(x_test, y_test[:,0])

    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = bs,
        shuffle = True,
        num_workers = 0,
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )
    testloader =  Data.DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )

    return trainloader, validloader, testloader

def split_train_valid_set(x_train, y_train, ratio):
    s = y_train.argsort()
    x_train = x_train[s]
    y_train = y_train[s]

    cL = int(len(x_train) / 4)

    class1_x = x_train[ 0 * cL : 1 * cL ]
    class2_x = x_train[ 1 * cL : 2 * cL ]
    class3_x = x_train[ 2 * cL : 3 * cL ]
    class4_x = x_train[ 3 * cL : 4 * cL ]

    class1_y = y_train[ 0 * cL : 1 * cL ]
    class2_y = y_train[ 1 * cL : 2 * cL ]
    class3_y = y_train[ 2 * cL : 3 * cL ]
    class4_y = y_train[ 3 * cL : 4 * cL ]

    vL = int(len(class1_x) / ratio)

    x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
    y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))

    x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
    y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))

    return x_train, y_train, x_valid, y_valid