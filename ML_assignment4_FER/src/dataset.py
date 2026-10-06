import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128, use_augmentation=False):
    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = base_transform

    train_dataset = datasets.ImageFolder(root='train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='test', transform=base_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, list(train_dataset.class_to_idx.keys())
