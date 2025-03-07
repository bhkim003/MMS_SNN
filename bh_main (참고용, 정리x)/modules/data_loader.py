import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import time

from snntorch import spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

from tqdm import tqdm

from apex.parallel import DistributedDataParallel as DDP

import random
import datetime


from torchvision import datasets, transforms
from sklearn.utils import shuffle

from PIL import Image

from torchtoolbox.transform import Cutout

''' 레퍼런스
https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.4/spikingjelly.datasets.html#module-spikingjelly.datasets
https://github.com/GorkaAbad/Sneaky-Spikes/blob/main/datasets.py
https://github.com/GorkaAbad/Sneaky-Spikes/blob/main/how_to.md
https://github.com/nmi-lab/torchneuromorphic
https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.spikedata.html#shd
'''

import snntorch
from snntorch.spikevision import spikedata

from modules.spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from modules.spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from modules.spikingjelly.datasets.n_mnist import NMNIST
# from modules.spikingjelly.datasets.es_imagenet import ESImageNet
from modules.spikingjelly.datasets import split_to_train_test_set
from modules.spikingjelly.datasets.n_caltech101 import NCaltech101
from modules.spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask

from typing import Callable, Dict, Optional, Tuple
import numpy as np
from modules.spikingjelly import datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from modules.spikingjelly import configure
from modules.spikingjelly.datasets import np_savez

import modules.torchneuromorphic.ntidigits.ntidigits_dataloaders as ntidigits_dataloaders

import pickle

import modules.tonic as tonic

from modules.tonic import DiskCachedDataset
import hashlib

from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *
from modules.ae_network import *

def data_loader(which_data, data_path, rate_coding, BATCH, IMAGE_SIZE, ddp_on, TIME, dvs_clipping, dvs_duration, exclude_class, merge_polarities, denoise_on, my_seed, extra_train_dataset, num_workers, chaching_on, pin_memory, train_data_split_indices):
    if (which_data == 'MNIST'):

        if rate_coding :
            transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0,), (1,))])
        else : 
            transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5))])

        trainset = torchvision.datasets.MNIST(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform)


        testset = torchvision.datasets.MNIST(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=transform)
        
        # torch.save((trainset.data, trainset.targets), train_data_path)
        # torch.save((testset.data, testset.targets), test_data_path)

        if (ddp_on == True):
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    num_workers =num_workers,
                                    sampler=train_sampler,
                                    pin_memory=pin_memory)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    num_workers =num_workers,
                                    sampler=test_sampler,
                                    pin_memory=pin_memory)
        else: 
            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    shuffle = True,
                                    num_workers =num_workers, generator=torch.Generator().manual_seed(my_seed),
                                    pin_memory=pin_memory)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    shuffle = False,
                                    num_workers =num_workers,
                                    pin_memory=pin_memory)
        synapse_conv_in_channels = 1
        CLASS_NUM = 10
        train_data_count = len(trainset) if train_data_split_indices == [] else len(train_data_split_indices)
        


    elif (which_data == 'CIFAR10'):

        if rate_coding :
            transform_train = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()])

            transform_test = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.ToTensor()])
        
        else :
            transform_train = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            transform_test = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])
                                            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            
            # # # ottt test
            # # print('지금 ottt만해라. 아니면 data_loader와서 바꿔라')
            # # assert IMAGE_SIZE == 32, 'OTTT랑 대조할 때는 32로 ㄱ'
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(IMAGE_SIZE, padding=4),
            #     Cutout(),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                         (0.2023, 0.1994, 0.2010)),
            # ])
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                         (0.2023, 0.1994, 0.2010)),
            # ])

        trainset = torchvision.datasets.CIFAR10(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform_train)


        testset = torchvision.datasets.CIFAR10(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
        
        
        if (ddp_on == True):
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    num_workers =num_workers,
                                    sampler=train_sampler,
                                    pin_memory=pin_memory)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    num_workers =num_workers,
                                    sampler=test_sampler,
                                    pin_memory=pin_memory)
        else: 
            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    shuffle = True,
                                    num_workers =num_workers, generator=torch.Generator().manual_seed(my_seed),
                                    pin_memory=pin_memory) # True, False
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    shuffle = False,
                                    num_workers =num_workers,
                                    pin_memory=pin_memory)
        synapse_conv_in_channels = 3
        CLASS_NUM = 10
        train_data_count = len(trainset) if train_data_split_indices == [] else len(train_data_split_indices)
        


        '''
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck') 
        '''



    # CIFAR100의 설정
    elif which_data == 'CIFAR100':
        if rate_coding:
            transform_train = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            transform_test = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor()
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

            transform_test = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

        if ddp_on:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

            train_loader = DataLoader(trainset, batch_size=BATCH, num_workers=num_workers, sampler=train_sampler,
                                    pin_memory=pin_memory)
            test_loader = DataLoader(testset, batch_size=BATCH, num_workers=num_workers, sampler=test_sampler,
                                    pin_memory=pin_memory)
        else:
            train_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(my_seed),
                                    pin_memory=pin_memory)
            test_loader = DataLoader(testset, batch_size=BATCH, shuffle=False, num_workers=num_workers,
                                    pin_memory=pin_memory)

        synapse_conv_in_channels = 3
        CLASS_NUM = 100

        '''
        classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
                'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
                'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
        '''
        train_data_count = len(trainset) if train_data_split_indices == [] else len(train_data_split_indices)


    elif (which_data == 'FASHION_MNIST'):

        if rate_coding :
            transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.ToTensor()])
        else : 
            transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5))])

        trainset = torchvision.datasets.FashionMNIST(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform)


        testset = torchvision.datasets.FashionMNIST(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=transform)

        if (ddp_on == True):
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    num_workers =num_workers,
                                    sampler=train_sampler,
                                    pin_memory=pin_memory)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    num_workers =num_workers,
                                    sampler=test_sampler,
                                    pin_memory=pin_memory)
        else: 
            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    shuffle = True,
                                    num_workers =num_workers, generator=torch.Generator().manual_seed(my_seed),
                                    pin_memory=pin_memory)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    shuffle = False,
                                    num_workers =num_workers,
                                    pin_memory=pin_memory)
        synapse_conv_in_channels = 1
        CLASS_NUM = 10
        train_data_count = len(trainset) if train_data_split_indices == [] else len(train_data_split_indices)




    # reference: H Zheng, Y Wu, L Deng, Y Hu, G Li. "Tdbn: Going Deeper with Directly-Trained Larger Spiking Neural Networks." Proceedings of the AAAI conference on artificial intelligence  (2021). Print.
    elif (which_data == 'DVS_CIFAR10'):
        data_path = data_path + '/cifar-dvs'
        train_path = data_path + '/train'
        val_path = data_path + '/test'
        train_dataset = DVSCifar10(path=train_path, transform=True, img_size = IMAGE_SIZE)
        val_dataset = DVSCifar10(path=val_path, transform=False, img_size = IMAGE_SIZE)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True,
                                                    num_workers=num_workers, pin_memory=pin_memory, generator=torch.Generator().manual_seed(my_seed))
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH, shuffle=False,
                                                    num_workers=num_workers, pin_memory=pin_memory)
        synapse_conv_in_channels = 2
        CLASS_NUM = 10
        train_data_count = len(train_dataset) if train_data_split_indices == [] else len(train_data_split_indices)




    # reference: Xiao, Mingqing, et al. "Hebbian learning based orthogonal projection for continual learning of spiking neural networks." arXiv preprint arXiv:2402.11984 (2024).
    elif (which_data == 'PMNIST'):
        assert False, 'PMNIST is not supported in this version'
        data, taskcla, size = pmnist_get(data_dir=data_path, seed=0, fixed_order=False)


        train_loader = data  # 다른 dataset과 다르니 외부에서 리빌딩해라
        test_loader = taskcla # 다른 dataset과 다르니 외부에서 리빌딩해라
        synapse_conv_in_channels = 1
        CLASS_NUM = 10



    elif (which_data == 'DVS_GESTURE'):
        assert False, '이거 쓰지말고 토닉 제스처 쓰자'
        # 쓰고 싶으면 옛날 교윤형님께 보낸 메일 확인해보자.
        data_dir = data_path + '/gesture'
        transform = None

        # # modules.spikingjelly.datasets.dvs128_gesture.DVS128Gesture(root: str, train: bool, use_frame=True, frames_num=10, split_by='number', normalization='max')

        #https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/neuromorphic_datasets.html
        # 10ms마다 1개의 timestep하고 싶으면 위의 주소 참고. 근데 timestep이 각각 좀 다를 거임.

        resize_shape = (IMAGE_SIZE, IMAGE_SIZE)
        if dvs_duration > 0:
            train_data = CustomDVS128Gesture(
                data_dir, train=True, data_type='frame',  split_by='time',  duration=dvs_duration, resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)
            test_data = CustomDVS128Gesture(
                data_dir, train=False, data_type='frame',  split_by='time',  duration=dvs_duration, resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)
        else:
            train_data = CustomDVS128Gesture(
                data_dir, train=True, data_type='frame', split_by='number', frames_number=TIME, resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)
            test_data = CustomDVS128Gesture(
                data_dir, train=False, data_type='frame', split_by='number', frames_number=TIME, resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)
        
        print(f'train samples = {train_data.__len__()}, test samples = {test_data.__len__()}')
        print(f'total samples = {train_data.__len__() + test_data.__len__()}')

        ## 'Other' 클래스 배제 ########################################################################
        # gesture_mapping = { 0 :'Hand Clapping' , 1 :'Right Hand Wave', 2:'Other',  3 :'Left Hand Wave' ,4 :'Right Arm CW'  , 5 :'Right Arm CCW' , 6 :'Left Arm CW' ,   7 :'Left Arm CCW' ,  8 :'Arm Roll'   ,    9 :'Air Drums'  ,    10 :'Air Guitar'}
        # 위의 맵핑을 보면 2번 클래스가 'Other'이다. 이 클래스를 배제하고 10개의 클래스만 사용하고 싶다.
        # class mapping = { 0 :'Hand Clapping'  1 :'Right Hand Wave'2 :'Left Hand Wave' 3 :'Right Arm CW'   4 :'Right Arm CCW'  5 :'Left Arm CW'    6 :'Left Arm CCW'   7 :'Arm Roll'       8 :'Air Drums'      9 :'Air Guitar'}
        # 그래서 이렇게 맵핑을 할 것임. 여기서는 'Other'만 배제시키고 train, test function에서 0~9까지 라벨을 세팅할 것임.
        # 참고로 arm roll이 앞으로 한번 뒤로 한번 감기 때문에 다른 data보다 두배로 많다.
        # 그리고 user2의 'lab'에서 hand clapping이 없고 user12의 'fluorescent led'에서 left arm ccw가 두개 있음
        
        # trainset
            # Class 8: 196 samples
            # Class 5: 98 samples
            # Class 9: 98 samples
            # Class 4: 98 samples
            # Class 3: 98 samples
            # Class 7: 99 samples
            # Class 6: 98 samples
            # Class 1: 98 samples
            # Class 0: 97 samples
            # Class 10: 98 samples

        # testset
            # Class 9: 24 samples
            # Class 5: 24 samples
            # Class 4: 24 samples
            # Class 8: 48 samples
            # Class 1: 24 samples
            # Class 10: 24 samples
            # Class 3: 24 samples
            # Class 0: 24 samples
            # Class 7: 24 samples
            # Class 6: 24 samples
        
        exclude_class = 2
        if dvs_duration > 0:
            train_file_name = f'{data_dir}/dvs_gesture_class_index/train_indices_dvsgesture_duration_{dvs_duration}'
            test_file_name = f'{data_dir}/dvs_gesture_class_index/test_indices_dvsgesture_duration_{dvs_duration}'
            if (os.path.isfile(train_file_name) and os.path.isfile(test_file_name)):
                print('\ndvsgesture 10 classes\' indices exist. we want to exclude the \'other\' class\n')
                with open(train_file_name, 'rb') as f:
                    train_indices = pickle.load(f)
                with open(test_file_name, 'rb') as f:
                    test_indices = pickle.load(f)
            else:
                print('\ndvsgesture 10 classes\' indices doesn\'t exist. we want to exclude the \'other\' class\n')
                train_indices = [i for i, (_, target) in enumerate(train_data) if target != exclude_class]
                test_indices = [i for i, (_, target) in enumerate(test_data) if target != exclude_class]
                with open(train_file_name, 'wb') as f:
                    pickle.dump(train_indices, f)
                with open(test_file_name, 'wb') as f:
                    pickle.dump(test_indices, f)
        else:
            train_indices = [i for i, (_, target) in enumerate(train_data) if target != exclude_class]
            test_indices = [i for i, (_, target) in enumerate(test_data) if target != exclude_class]
        ################################################################################################

        # SubsetRandomSampler 생성
        train_sampler = SubsetRandomSampler(train_indices, generator=torch.Generator().manual_seed(my_seed))
        test_sampler = SubsetRandomSampler(test_indices) # 원래 SequentialSampler써야 되는 게 맞는데 그러면 배제시킨 2-class에 다른 게 채워짐. 그래서 그냥 SubsetRandomSampler씀.

        # ([B, T, 2, 128, 128]) 
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH, num_workers=num_workers, sampler=train_sampler, collate_fn=pad_sequence_collate, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH, num_workers=num_workers, sampler=test_sampler, collate_fn=pad_sequence_collate, pin_memory=pin_memory)
        synapse_conv_in_channels = 2
        CLASS_NUM = 10
        train_data_count = len(train_indices) if train_data_split_indices == [] else len(train_data_split_indices)

        
    elif (which_data == 'DVS_GESTURE_TONIC'):
        data_dir = data_path
        
        train_dataset_list = []
        for extra_train_index in range(extra_train_dataset+1):
            
            train_compose = []
            if merge_polarities == True:
                train_compose.append(tonic.transforms.MergePolarities()) #polarity 없애기
            # train_compose.append(tonic.transforms.CropTime(max=6_000_000))
            train_compose.append(tonic.transforms.CropTime(max=(100_000 + (2_300_000//(extra_train_dataset+1))*extra_train_index + dvs_duration*(TIME+1))))
            train_compose.append(tonic.transforms.CropTime(min=(100_000 + (2_300_000//(extra_train_dataset+1))*extra_train_index)))
            if denoise_on == True:
                train_compose.append(tonic.transforms.Denoise(filter_time=10_000)) # 10_000 # 낮을수록 더 많이 거름
            train_compose.append(tonic.transforms.Downsample(spatial_factor=IMAGE_SIZE/tonic.datasets.DVSGesture.sensor_size[0]))
            train_compose.append(tonic.transforms.ToFrame(
                sensor_size=(IMAGE_SIZE,IMAGE_SIZE,2),
                time_window=dvs_duration, 
                include_incomplete=False))
            train_transform = tonic.transforms.Compose(train_compose)

            test_compose = []
            if merge_polarities == True:
                test_compose.append(tonic.transforms.MergePolarities()) #polarity 없애기
            test_compose.append(tonic.transforms.CropTime(max=100_000 + dvs_duration*(TIME+1)))
            test_compose.append(tonic.transforms.CropTime(min=100_000))
            if denoise_on == True:
                test_compose.append(tonic.transforms.Denoise(filter_time=10_000)) # 10_000 # 낮을수록 더 많이 거름
            test_compose.append(tonic.transforms.Downsample(spatial_factor=IMAGE_SIZE/tonic.datasets.DVSGesture.sensor_size[0]))
            test_compose.append(tonic.transforms.ToFrame(
                sensor_size=(IMAGE_SIZE,IMAGE_SIZE,2),
                time_window=dvs_duration, 
                include_incomplete=False))
            test_transform = tonic.transforms.Compose(test_compose)


            train_dataset_temp = tonic.datasets.DVSGesture(data_dir, train=True, transform=train_transform, clipping = dvs_clipping, time = TIME, exclude_class = exclude_class)
            test_dataset = tonic.datasets.DVSGesture(data_dir, train=False, transform=test_transform, clipping = dvs_clipping, time = TIME, exclude_class = exclude_class)
        

            ## disk에 dataset caching하기 ###################################################################
            train_transform_settings = {
                'train_transform': train_transform,
                'test_transform': test_transform,
                'clipping': dvs_clipping,
                'time': TIME,
                'exclude_class': exclude_class,
            }
            settings_str = str(train_transform_settings)
            dataset_hash = hashlib.md5(settings_str.encode()).hexdigest()
            print(f'dataset_hash = {dataset_hash}')

            my_cache_path = f"{data_dir}/DVSGesture/cache/fast_dataloading_{dataset_hash}"
            if os.path.exists(my_cache_path):
                print('cache path exists')
            else:
                print('cache path doesn\'t exist')
                os.mkdir(my_cache_path)
                os.mkdir(f"{my_cache_path}/train")
                os.mkdir(f"{my_cache_path}/test")

            if chaching_on == True:
                ## for extra train set ##################################################################
                train_dataset_list.append(DiskCachedDataset(train_dataset_temp, cache_path=f"{my_cache_path}/train"))
                #########################################################################################
                test_dataset = DiskCachedDataset(test_dataset, cache_path=f"{my_cache_path}/test")
                ################################################################################################
            else: 
                ## for extra train set ##################################################################
                train_dataset_list.append(train_dataset_temp)
                #########################################################################################
                test_dataset = test_dataset
                ################################################################################################
        
        train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
        if train_data_split_indices == []: # no split
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle = True, num_workers=num_workers, drop_last=False, generator=torch.Generator().manual_seed(my_seed), pin_memory = pin_memory)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, sampler=SubsetRandomSampler(train_data_split_indices, generator=torch.Generator().manual_seed(my_seed)), num_workers=num_workers, drop_last=False, pin_memory = pin_memory)
            
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, shuffle = False, num_workers=num_workers, drop_last=False, pin_memory = pin_memory)
        synapse_conv_in_channels = 1 if merge_polarities == True else 2
        CLASS_NUM = 10 if exclude_class == True else 11

        train_data_count = len(train_dataset) if train_data_split_indices == [] else len(train_data_split_indices)
        ###################################################################################################

    elif (which_data == 'DVS_CIFAR10_2'): # 느림
        data_dir = data_path + '/cifar10-dvs2/cifar10'

        # Split by number as in: https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron
        # classspikingjelly.datasets.cifar10_dvs.CIFAR10DVS(root: str, train: bool, split_ratio=0.9, use_frame=True, frames_num=10, split_by='number', normalization='max')
        
        if dvs_duration > 0:
            resize_shape = (IMAGE_SIZE, IMAGE_SIZE)
            dataset = CustomCIFAR10DVS(data_dir, data_type='frame',  split_by='time', duration=dvs_duration,
                                resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)
            NAME = dvs_duration
        else: 
            dataset = CustomCIFAR10DVS(data_dir, data_type='frame',
                                split_by='number', frames_number=TIME, resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)
            NAME = TIME

        path_train = os.path.join(data_dir, f'{NAME}_{IMAGE_SIZE}_train_split.pt')
        path_test = os.path.join(data_dir, f'{NAME}_{IMAGE_SIZE}_test_split.pt')
        if os.path.exists(path_train) and os.path.exists(path_test):
            train_set = torch.load(path_train)
            test_set = torch.load(path_test)
        else:
            train_set, test_set = split_to_train_test_set(
                origin_dataset=dataset, train_ratio=0.9, num_classes=10)

            torch.save(train_set, path_train)
            torch.save(test_set, path_test)
        
        # ([B, T, 2, 128, 128])
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(my_seed), pin_memory = pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory = pin_memory)
        synapse_conv_in_channels = 2
        CLASS_NUM = 10
        train_data_count = len(train_set) if train_data_split_indices == [] else len(train_data_split_indices)



    elif (which_data == 'NMNIST'):
        data_dir = data_path + '/nmnist/mnist'
        # spikingjelly.datasets.n_mnist.NMNIST(root: str, train: bool, use_frame=True, frames_num=10, split_by='number', normalization='max')

        resize_shape = (IMAGE_SIZE, IMAGE_SIZE)

        if dvs_duration > 0:
            train_set = CustomNMNIST(data_dir, train=True, data_type='frame',  split_by='time', duration=dvs_duration,
                                resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)

            test_set = CustomNMNIST(data_dir, train=False, data_type='frame',  split_by='time', duration=dvs_duration,
                            resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)
        else: 
            train_set = CustomNMNIST(data_dir, train=True, data_type='frame',
                            split_by='number', frames_number=TIME, resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)

            test_set = CustomNMNIST(data_dir, train=False, data_type='frame',
                            split_by='number', frames_number=TIME, resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)


        # ([B, T, 2, 34, 34])
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(my_seed), pin_memory = pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory = pin_memory)
        synapse_conv_in_channels = 2
        CLASS_NUM = 10
        train_data_count = len(train_set) if train_data_split_indices == [] else len(train_data_split_indices)

    elif (which_data == 'NMNIST_TONIC'):
        data_dir = data_path 
        
        compose = []
        if merge_polarities == True:
            compose.append(tonic.transforms.MergePolarities()) #polarity 없애기
        compose.append(tonic.transforms.CropTime(max=(10_000 + dvs_duration*(TIME+1))+100_000))
        compose.append(tonic.transforms.CropTime(min=(10_000)))
        if denoise_on == True:
            compose.append(tonic.transforms.Denoise(filter_time=10_000)) # 10_000 # 낮을수록 더 많이 거름
        compose.append(tonic.transforms.Downsample(spatial_factor=IMAGE_SIZE/tonic.datasets.NMNIST.sensor_size[0]))
        compose.append(tonic.transforms.ToFrame(
            sensor_size=(IMAGE_SIZE,IMAGE_SIZE,2),
            time_window=dvs_duration, 
            include_incomplete=False))
        transform = tonic.transforms.Compose(compose)



        train_dataset = tonic.datasets.NMNIST(data_dir, train=True, transform=transform, clipping = dvs_clipping, time = TIME)
        test_dataset = tonic.datasets.NMNIST(data_dir, train=False, transform=transform, clipping = dvs_clipping, time = TIME)
        

        ## disk에 dataset caching하기 ###################################################################
        transform_settings = {
            'train_transform': transform,
            'test_transform': transform,
            'clipping': dvs_clipping,
            'time': TIME
        }
        settings_str = str(transform_settings)
        dataset_hash = hashlib.md5(settings_str.encode()).hexdigest()
        print(f'dataset_hash = {dataset_hash}')

        my_cache_path = f"{data_dir}/NMNIST/cache/fast_dataloading_{dataset_hash}"
        if os.path.exists(my_cache_path):
            print('cache path exists')
        else:
            print('cache path doesn\'t exist')
            os.mkdir(my_cache_path)
            os.mkdir(f"{my_cache_path}/train")
            os.mkdir(f"{my_cache_path}/test")

        if chaching_on == True:
            train_dataset = DiskCachedDataset(train_dataset, cache_path=f"{my_cache_path}/train")
            test_dataset = DiskCachedDataset(test_dataset, cache_path=f"{my_cache_path}/test")
        else:
            train_dataset = train_dataset
            test_dataset = test_dataset
        ################################################################################################



        if train_data_split_indices == []: # no split
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle = True, num_workers=num_workers, drop_last=False, generator=torch.Generator().manual_seed(my_seed), pin_memory = pin_memory)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, sampler=SubsetRandomSampler(train_data_split_indices, generator=torch.Generator().manual_seed(my_seed)), num_workers=num_workers, drop_last=False, pin_memory = pin_memory)
            
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, shuffle = False, num_workers=num_workers, drop_last=False, pin_memory = pin_memory)

        if merge_polarities == True:
            synapse_conv_in_channels = 1
        else:  
            synapse_conv_in_channels = 2
        CLASS_NUM = 10
        train_data_count = len(train_dataset) if train_data_split_indices == [] else len(train_data_split_indices)




    elif (which_data == 'N_CALTECH101'):
        data_dir = data_path + '/ncaltech/caltech'
        path_train = os.path.join(data_dir, f'{TIME}_{IMAGE_SIZE}_train_split.pt')
        path_test = os.path.join(data_dir, f'{TIME}_{IMAGE_SIZE}_test_split.pt')

        #root: str, data_type: str = 'event', frames_number: int = None, split_by: str = None, duration: int = None, custom_integrate_function: Callable = None, custom_integrated_frames_dir_name: str = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
        resize_shape = (IMAGE_SIZE, IMAGE_SIZE)
        dataset = CustomNCaltech101(data_dir, data_type='frame',
                            split_by='number', frames_number=TIME, resize_shape=resize_shape, dvs_clipping=dvs_clipping, dvs_duration_copy=dvs_duration, TIME=TIME)

        if os.path.exists(path_train) and os.path.exists(path_test):
            train_set = torch.load(path_train)
            test_set = torch.load(path_test)
        else:
            train_set, test_set = split_to_train_test_set(
                origin_dataset=dataset, train_ratio=0.9, num_classes=101)

            torch.save(train_set, path_train)
            torch.save(test_set, path_test)

        # ([B, T, 2, 180, 240])
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(my_seed), pin_memory = pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory = pin_memory)
        synapse_conv_in_channels = 2
        CLASS_NUM = 100
        train_data_count = len(train_set) if train_data_split_indices == [] else len(train_data_split_indices)




    elif which_data == 'n_tidigits':
        data_dir = data_path + '/ntidigits/ntidigits_isolated.hdf5'

        # root = 'data/tidigits/ntidigits_isolated.hdf5', batch_size = 72 , chunk_size_train = 1000, chunk_size_test = 1000, ds = 1, dt = 1000, transform_train = None, transform_test = None, target_transform_train = None, target_transform_test = None, **dl_kwargs):
        train_loader, test_loader = ntidigits_dataloaders.create_dataloader(
            root = data_dir, chunk_size_train=1000, chunk_size_test=1000, batch_size=BATCH, dt = 1000, ds = [1], num_workers=num_workers)
        synapse_conv_in_channels = 64 # conv inchannel이 아니고 FC in_channel
        CLASS_NUM = 11
        # mapping = { 0 :'0',
        #     1 :'1',
        #     2 :'2',
        #     3 :'3',
        #     4 :'4',
        #     5 :'5',
        #     6 :'6',
        #     7 :'7',
        #     8 :'8',
        #     9 :'9',
        #     10: '10'}
        assert False, "이거 train_data_count 몇개임"
        train_data_count = len(train_set) if train_data_split_indices == [] else len(train_data_split_indices)


    elif which_data == 'heidelberg':
        data_dir = data_path + '/Heidelberg'

        # root, train=True, transform=None, target_transform=None, download_and_create=True, num_steps=1000, ds=1, dt=1000)
        train_ds = spikedata.SHD(data_dir, train=True)
        test_ds = spikedata.SHD(data_dir, train=False)

        # create dataloaders
        train_loader = DataLoader(dataset=train_ds, batch_size=BATCH, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(my_seed)) # 8156x2x1000x700
        test_loader = DataLoader(dataset=test_ds, batch_size=BATCH, shuffle=False, num_workers=num_workers) # 2264x2x1000x700
        synapse_conv_in_channels = 700 # conv inchannel이 아니고 FC in_channel
        CLASS_NUM = 20
        ''' 한번프린트해보기
        import snntorch as snn
        from snntorch.spikevision import spikedata
        from torch.utils.data import DataLoader


        # root, train=True, transform=None, target_transform=None, download_and_create=True, num_steps=1000, ds=1, dt=1000)
        train_ds = spikedata.SHD("/data2/Heidelberg", train=True)
        test_ds = spikedata.SHD("/data2/Heidelberg", train=False)

        # create dataloaders
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=64) # 8156x2x1000x700
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=64) # 2264x2x1000x700

        import matplotlib.pyplot as plt
        import snntorch.spikeplot as splt

        # choose a random sample
        n = 6295

        # initialize figure and axes
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)

        # use spikeplot to generate a raster
        splt.raster(train_dl.dataset[n][0], ax, s=1.5, c="black")
        '''
        train_data_count = len(train_ds) if train_data_split_indices == [] else len(train_data_split_indices)



    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    return train_loader, test_loader, synapse_conv_in_channels, CLASS_NUM, train_data_count







###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################






# reference: H Zheng, Y Wu, L Deng, Y Hu, G Li. "Tdbn: Going Deeper with Directly-Trained Larger Spiking Neural Networks." Proceedings of the AAAI conference on artificial intelligence  (2021). Print.
class DVSCifar10(Dataset):
    def __init__(self, path, transform=None, img_size = 48):
        self.path = path
        self.transform = transform
        self.img_size = img_size

        # 일단 기본적인 resize는 무조건 해주고
        # rotate하고 shear(x방향)하고 roll은 무작위로함.
        self.resize = transforms.Resize(size=(self.img_size, self.img_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))
        self.cutout = Cutout_NDA(length=16)  # Cutout length (patch size) set to 16 pixels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.path + '/{}.pt'.format(index))
        # print('원래사이즈', data.size())
        data = self.resize(data.permute([3, 0, 1, 2]))
        if self.transform == True:
            choices = ['roll', 'rotate', 'shear', 'cutout']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            elif aug == 'rotate':
                data = self.rotate(data)
            elif aug == 'shear':
                data = self.shearx(data)
            elif aug == 'cutout':
                data = self.cutout(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.path))
    
class Cutout_NDA(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


# reference: Xiao, Mingqing, et al. "Hebbian learning based orthogonal projection for continual learning of spiking neural networks." arXiv preprint arXiv:2402.11984 (2024).
def pmnist_get(data_dir='/data2', seed=0, fixed_order=False):
    data = {}
    taskcla = []
    size = [1, 28, 28]

    mnist_dir = data_dir
    pmnist_dir = os.path.join(mnist_dir, 'binary_pmnist')

    nperm = 10  # 10 tasks
    seeds = np.array(list(range(nperm)), dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds, random_state=seed)

    if not os.path.isdir(pmnist_dir):
        os.makedirs(pmnist_dir)
        # Pre-load
        # MNIST
        mean = (0.1307,)
        std = (0.3081,)
        dat = {}
        dat['train'] = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.MNIST(mnist_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for i, r in enumerate(seeds):
            print(i, end=',')
            sys.stdout.flush()
            data[i] = {}
            data[i]['name'] = 'pmnist-{:d}'.format(i)
            data[i]['ncla'] = 10
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[i][s] = {'x': [], 'y': []}
                for image, target in loader:
                    aux = image.view(-1).numpy()
                    aux = shuffle(aux, random_state=r * 100 + i)
                    image = torch.FloatTensor(aux).view(size)
                    data[i][s]['x'].append(image)
                    data[i][s]['y'].append(target.numpy()[0])

            # "Unify" and save
            for s in ['train', 'test']: # test, train 각각 x,y를 저장
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))
        print()

    else:

        # Load binary files
        for i, r in enumerate(seeds):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 10
            data[i]['name'] = 'pmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))

    # Validation
    #for t in data.keys():
    #    r=np.arange(data[t]['train']['x'].size(0))
    #    # r=np.array(shuffle(r,random_state=seed),dtype=int)
    #    r=np.array(r,dtype=int)
    #    nvalid=int(pc_valid*len(r))
    #    ivalid=torch.LongTensor(r[:nvalid])
    #    itrain=torch.LongTensor(r[nvalid:])
    #    data[t]['valid'] = {}
    #    data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
    #    data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
    #    data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
    #    data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size



####################################################################################################################################
def numpy_to_pil(img):
    img = img.transpose(1, 2, 0)  # (2, 128, 128) -> (128, 128, 2)
    return Image.fromarray(img.astype(np.uint8))


# 커스텀 데이터셋 클래스
class CustomDVS128Gesture(DVS128Gesture):
    def __init__(self, *args, resize_shape=(128, 128), dvs_clipping = 1, dvs_duration_copy=1000000, TIME=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize_shape = resize_shape
        self.resize_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(self.resize_shape ),
        ])
        self.dvs_clipping = dvs_clipping
        self.dvs_duration_copy = dvs_duration_copy
        self.TIME = TIME
        # self.resize_transform = transforms.Resize(self.resize_shape, interpolation=transforms.InterpolationMode.NEAREST)
    def __getitem__(self, index):
        # 원본 데이터를 가져옵니다
        data, target = super().__getitem__(index)
        # data는 numpy array 형태입니다. (T, 2, 128, 128)
        # print('data_size', data.shape)


        # 각 프레임을 PIL 이미지로 변환 후 리사이즈합니다
        resized_frames = []
        for frame in data:
            pil_img = numpy_to_pil(frame)
            resized_img = self.resize_transform(pil_img)
            resized_frames.append(np.array(resized_img).transpose(2, 0, 1)) # (128, 128, 2) -> (2, 128, 128)
            
        resized_data = np.stack(resized_frames) # (T, 2, 128, 128)
        resized_data = torch.tensor(resized_data, dtype=torch.float32)  # torch.float32로 변환

        if self.dvs_clipping != 0:
            resized_data[resized_data < self.dvs_clipping] = 0.0
            resized_data[resized_data >= self.dvs_clipping] = 1.0
            # ANP-I에서는 4개 스파이크 모이면 1로 했음.
            # 너도 그럴려면 위에 transforms.Compose에서 transform.ToTensor빼고 여기서 4이상인 건 1, 그 외 0으로 ㄱㄱ

        # resized_data = resized_data.permute(0,2,3,1)

        # 시간단위로 샘플링 했을 때 TIME으로 맞추기
        if (self.dvs_duration_copy > 0):
            # print('resized_data', resized_data.size())
            T, *spatial_dims = resized_data.shape
            if T > self.TIME:
                resized_data = resized_data[:self.TIME]
            else:
                resized_data = torch.cat([resized_data, torch.zeros(self.TIME - T, *spatial_dims)], dim=0)
        # print(resized_data.size())
        return resized_data, target
    



# 커스텀 데이터셋 클래스
class CustomCIFAR10DVS(CIFAR10DVS):
    def __init__(self, *args, resize_shape=(128, 128), dvs_clipping = 1, dvs_duration_copy=30000, TIME=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize_shape = resize_shape
        self.resize_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(self.resize_shape ),
        ])
        self.dvs_clipping = dvs_clipping
        self.dvs_duration_copy = dvs_duration_copy
        self.TIME = TIME
        # self.resize_transform = transforms.Resize(self.resize_shape, interpolation=transforms.InterpolationMode.NEAREST)
    def __getitem__(self, index):
        # 원본 데이터를 가져옵니다
        data, target = super().__getitem__(index)
        # data는 numpy array 형태입니다. (T, 2, 128, 128)

        # 각 프레임을 PIL 이미지로 변환 후 리사이즈합니다
        resized_frames = []
        for frame in data:
            pil_img = numpy_to_pil(frame)
            resized_img = self.resize_transform(pil_img)
            k = np.array(resized_img)
            resized_frames.append(np.array(resized_img).transpose(2, 0, 1))# (128, 128, 2) -> (2, 128, 128)
            
        resized_data = np.stack(resized_frames)  # (T, 2, 128, 128)
        resized_data = torch.tensor(resized_data, dtype=torch.float32)  # torch.float32로 변환

        if self.dvs_clipping != 0:
            resized_data[resized_data < self.dvs_clipping] = 0.0
            resized_data[resized_data >= self.dvs_clipping] = 1.0
        resized_data = resized_data.permute(0,2,3,1)

        # 시간단위로 샘플링 했을 때 TIME으로 맞추기
        if (self.dvs_duration_copy > 0):
            T, *spatial_dims = resized_data.shape
            if T > self.TIME:
                resized_data = resized_data[:self.TIME]
            else:
                resized_data = torch.cat([resized_data, torch.zeros(self.TIME - T, *spatial_dims)], dim=0)
        return resized_data, target
    
    
# 커스텀 데이터셋 클래스
class CustomNMNIST(NMNIST):
    def __init__(self, *args, resize_shape=(34, 34), dvs_clipping = 1, dvs_duration_copy=30000, TIME=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize_shape = resize_shape
        
        self.resize_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(self.resize_shape ),
        ])
        self.dvs_clipping = dvs_clipping
        self.dvs_duration_copy = dvs_duration_copy
        self.TIME = TIME
        # self.resize_transform = transforms.Resize(self.resize_shape, interpolation=transforms.InterpolationMode.NEAREST)
    def __getitem__(self, index):
        # 원본 데이터를 가져옵니다
        data, target = super().__getitem__(index)
        # data는 numpy array 형태입니다. (T, 2, 128, 128)

        # 각 프레임을 PIL 이미지로 변환 후 리사이즈합니다
        resized_frames = []
        for frame in data:
            pil_img = numpy_to_pil(frame)
            resized_img = self.resize_transform(pil_img) 
            resized_img = np.array(resized_img)
            resized_frames.append(np.array(resized_img).transpose(2, 0, 1))# (128, 128, 2) -> (2, 128, 128)
            
        resized_data = np.stack(resized_frames)  # (T, 2, 128, 128)
        resized_data = torch.tensor(resized_data, dtype=torch.float32)  # torch.float32로 변환

        if self.dvs_clipping != 0:
            resized_data[resized_data < self.dvs_clipping] = 0.0
            resized_data[resized_data >= self.dvs_clipping] = 1.0

        # resized_data = resized_data.permute(0,2,3,1)


        # 시간단위로 샘플링 했을 때 TIME으로 맞추기
        if (self.dvs_duration_copy > 0):
            T, *spatial_dims = resized_data.shape
            if T > self.TIME:
                resized_data = resized_data[:self.TIME]
            else:
                resized_data = torch.cat([resized_data, torch.zeros(self.TIME - T, *spatial_dims)], dim=0)

        # print('in_custom',resized_data.size()) # T, 2, 128,128이 떠야함.
        return resized_data, target
    

# 커스텀 데이터셋 클래스
class CustomNCaltech101(NCaltech101):
    def __init__(self, *args, resize_shape=(180, 240), dvs_clipping = 1, dvs_duration_copy=30000, TIME=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize_shape = resize_shape
        self.resize_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(self.resize_shape ),
        ])
        self.dvs_clipping = dvs_clipping
        self.dvs_duration_copy = dvs_duration_copy
        self.TIME = TIME
        # self.resize_transform = transforms.Resize(self.resize_shape, interpolation=transforms.InterpolationMode.NEAREST)
    def __getitem__(self, index):
        # 원본 데이터를 가져옵니다
        data, target = super().__getitem__(index)
        # data는 numpy array 형태입니다. (T, 2, 128, 128)

        # 각 프레임을 PIL 이미지로 변환 후 리사이즈합니다
        resized_frames = []
        for frame in data:
            pil_img = numpy_to_pil(frame)
            resized_img = self.resize_transform(pil_img)
            k = np.array(resized_img)
            resized_frames.append(np.array(resized_img).transpose(2, 0, 1))# (128, 128, 2) -> (2, 128, 128)
            
        resized_data = np.stack(resized_frames)  # (T, 2, 128, 128)
        resized_data = torch.tensor(resized_data, dtype=torch.float32)  # torch.float32로 변환
        
        if self.dvs_clipping != 0:
            resized_data[resized_data < self.dvs_clipping] = 0.0
            resized_data[resized_data >= self.dvs_clipping] = 1.0

        # resized_data = resized_data.permute(0,2,3,1)

        # 시간단위로 샘플링 했을 때 TIME으로 맞추기
        if (self.dvs_duration_copy > 0):
            T, *spatial_dims = resized_data.shape
            if T > self.TIME:
                resized_data = resized_data[:self.TIME]
            else:
                resized_data = torch.cat([resized_data, torch.zeros(self.TIME - T, *spatial_dims)], dim=0)
        return resized_data, target
    


def plot_2d_array(array, title='2D Array Plot', cmap='viridis'):
    """
    2차원 NumPy 배열을 플로팅하는 함수.

    Parameters:
    - array: 2차원 NumPy 배열
    - title: 플롯 제목 (기본값: '2D Array Plot')
    - cmap: 색상 맵 (기본값: 'viridis')

    Returns:
    - None
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    plt.figure(figsize=(8, 6))
    plt.imshow(array, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid(False)
    plt.show()

# 예시 사용
array = np.random.rand(10, 10)  # 임의의 2차원 배열 생성
plot_2d_array(array, title='Random 2D Array')