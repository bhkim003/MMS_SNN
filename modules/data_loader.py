import torch
import torchvision

import modules

def data_loader(which_data, data_path, batch_size):
    if (which_data == 'MNIST'):
        IMAGE_SIZE = 28 # MINST는 일반적으로 28

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 예: 32x32로 크기 변경
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = torchvision.datasets.MNIST(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform)

        testset = torchvision.datasets.MNIST(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset,
                                batch_size =batch_size,
                                shuffle = True,)
        test_loader = torch.utils.data.DataLoader(testset,
                                batch_size =batch_size,
                                shuffle = False,)
        
        CLASS_NUM = 10 # MNIST는 class_num
        in_channels = 1 # MNIST는 in_channel 1
        IMAGE_SIZE = IMAGE_SIZE
        return train_loader, test_loader, CLASS_NUM, in_channels, IMAGE_SIZE
