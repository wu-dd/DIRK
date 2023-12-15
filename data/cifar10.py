import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from partial_models.wide_resnet import WideResNet
from augment.randaugment import RandomAugment
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy
from utils.util import generate_instancedependent_candidate_labels


def load_cifar10(args):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    original_train = dsets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
    ori_data, ori_labels = original_train.data, torch.Tensor(original_train.targets).long()
    
    test_dataset = dsets.CIFAR10(root='./data/CIFAR10', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, \
        shuffle=False, num_workers=4, pin_memory=False
    )
    
    ori_data = torch.Tensor(original_train.data)
    model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
    model.load_state_dict(torch.load(os.path.expanduser('./partial_models/weights/cifar10.pt')))
    ori_data = ori_data.permute(0, 3, 1, 2)
    partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels,args.rate)
    ori_data = original_train.data

    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')
    
    partial_training_dataset = CIFAR10_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=20,
        pin_memory=True,
        drop_last=True
    )
    
    return partial_training_dataloader, partialY_matrix, test_loader


class CIFAR10_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        self.distill_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.ToPILImage(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        
        each_image_w = self.weak_transform(self.ori_images[index])
        each_image_s = self.strong_transform(self.ori_images[index])
        each_image_distill= self.distill_transform(self.ori_images[index])
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_image_distill, each_label, each_true_label, index

