# src/elephant_vision/reid_dataset.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_reid_dataloader(data_dir="artifacts/datasets/reid/train", batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return loader, dataset
